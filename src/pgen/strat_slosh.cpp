//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file strat_slosh.cpp
//! \brief Problem generator for stratified 3D shearing sheet with sloshing forcing.
//!
//! PURPOSE:  Problem generator for stratified 3D shearing sheet.  Based on the
//!   initial conditions described in "Three-dimensional Magnetohydrodynamic
//!   Simulations of Vertically Stratified Accretion Disks" by Stone, Hawley,
//!   Gammie & Balbus.
//!
//! Several different field configurations and perturbations are possible:
//! - ifield = 1 - Bz=B0 sin(x1) field with zero-net-flux [default]
//! - ifield = 2 - uniform Bz
//! - ifield = 3 - uniform Bz plus sinusoidal perturbation Bz(1+0.5*sin(kx*x1))
//! - ifield = 4 - B=(0,B0cos(kx*x1),B0sin(kx*x1))= zero-net flux w helicity
//! - ifield = 5 - uniform By, but only for |z|<2
//! - ifield = 6 - By with constant beta versus z
//! - ifield = 7 - zero field everywhere
//! - ifield = 8 - divergence-free random noise field (curl of random vector potential)
//!
//! - ipert = 1 - random perturbations to P and V [default, used by HGB]
//! ~ iv    = 0 - keplerian velocity only
//! ~ iv    = 1 - keplerian velocity + initial slosh mode velocity
//!
//! Sloshing forcing parameters:
//! - psi   : forcing amplitude (dimensionless)
//! - rise_t: exponential rise timescale; 0 = instantaneous onset [default]
//!
//! Code must be configured using -shear
//!
//! REFERENCE:
//! - Stone, J., Hawley, J., Gammie, C.F. & Balbus, S. A., ApJ 463, 656-673 (1996)
//! - Hawley, J. F. & Balbus, S. A., ApJ 400, 595-609 (1992)
//============================================================================

// C headers

// C++ headers
#include <algorithm>
#include <cmath>      // sqrt()
#include <functional> // hash
#include <iostream>
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <random>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"     // ran2()

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void StratOutflowInnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void StratOutflowOuterX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);
namespace {
Real HistoryBxBy(MeshBlock *pmb, int iout);
Real HistorydVxVy(MeshBlock *pmb, int iout);

Real dfloor, pfloor;
Real Omega_0, qshear, psi_f, rise_t, Hbeta;
} // namespace

//====================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  qshear = pin->GetReal("orbital_advection","qshear");
  Omega_0 = pin->GetReal("orbital_advection","Omega0");
  psi_f   = pin->GetReal("problem","psi");
  rise_t  = pin->GetOrAddReal("problem","rise_t", 0.0);

  Real beta_val = 0.0;
  if (MAGNETIC_FIELDS_ENABLED) {
    beta_val = pin->GetReal("problem","beta");
    Hbeta = std::sqrt(1.0 + 1.0/beta_val);
  } else {
    Hbeta = 1.0;
  }

  AllocateUserHistoryOutput(2);
  if (MAGNETIC_FIELDS_ENABLED) {
    EnrollUserHistoryOutput(0, HistoryBxBy, "-BxBy");
  }
  EnrollUserHistoryOutput(1, HistorydVxVy, "dVxVy");

  EnrollUserExplicitSourceFunction(VertGrav);

  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, StratOutflowInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, StratOutflowOuterX3);
  }

  if (!shear_periodic) {
    std::stringstream msg;
    msg << "### FATAL ERROR in strat_slosh.cpp ProblemGenerator" << std::endl
        << "This problem generator requires shearing box." << std::endl;
    ATHENA_ERROR(msg);
  }

  return;
}



//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief stratified disk problem generator for 3D problems.
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int ifield, ipert, iv;
  Real beta, amp, pres;
  Real iso_cs=1.0;
  Real B0 = 0.0;

  Real SumRd=0.0, SumRvx=0.0, SumRvy=0.0, SumRvz=0.0;
  Real x1, x3;
  Real rd(0.0), rp(0.0);
  Real rvx, rvy, rvz;
  Real rval;

  const Real den=1.0;

  Real Lx = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  Real Lz = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;

  int nwx = pin->GetOrAddInteger("problem","nwx",1);
  Real kx = (2.0*PI/Lx)*(static_cast<Real>(nwx));

  std::int64_t iseed = -1 - gid;

  Real gam = peos->GetGamma();

  if (pmy_mesh->mesh_size.nx3 == 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in strat_slosh.cpp ProblemGenerator" << std::endl
        << "Stratified shearing sheet only works on a 3D grid" << std::endl;
    ATHENA_ERROR(msg);
  }

  amp = pin->GetReal("problem","amp");
  ipert = pin->GetOrAddInteger("problem","ipert", 1);
  iv = pin->GetOrAddInteger("problem","iv", 0);

  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(float_min)));

  if (MAGNETIC_FIELDS_ENABLED) {
    ifield = pin->GetOrAddInteger("problem","ifield", 1);
    beta = pin->GetReal("problem","beta");
  }

  if (NON_BAROTROPIC_EOS) {
    pres  = pin->GetOrAddReal("problem","pres",1.0);
  } else {
    iso_cs = peos->GetIsoSoundSpeed();
    pres = den*SQR(iso_cs);
  }

  // Allocate vector potential arrays for ifield=8 (noise field)
  int nx1 = block_size.nx1 + 2*NGHOST;
  int nx2 = block_size.nx2 + 2*NGHOST;
  int nx3 = block_size.nx3 + 2*NGHOST;
  AthenaArray<Real> a1, a2, a3;
  a1.NewAthenaArray(nx3, nx2, nx1);
  a2.NewAthenaArray(nx3, nx2, nx1);
  a3.NewAthenaArray(nx3, nx2, nx1);

  if (MAGNETIC_FIELDS_ENABLED) {
    B0 = std::sqrt(static_cast<Real>(2.0*pres/beta));
    if (ifield == 8) {
      // Divergence-free noise field via curl of random vector potential.
      // Use coordinate-based hashing so faces shared between meshblocks
      // get identical vector potential values, ensuring div(B)=0 globally.
      Real noise_amp = B0*std::pow(1.0/SQR(pcoord->dx1f(is))
                                 + 1.0/SQR(pcoord->dx2f(js))
                                 + 1.0/SQR(pcoord->dx3f(ks)), -0.5);
      std::hash<double> hasher;
      for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je+1; j++) {
          for (int i=is; i<=ie+1; i++) {
            Real xf1 = pcoord->x1f(i);
            Real xf2 = pcoord->x2f(j);
            Real xf3 = pcoord->x3f(k);
            Real xv1 = pcoord->x1v(i);
            Real xv2 = pcoord->x2v(j);
            Real xv3 = pcoord->x3v(k);
            // A1 at (x1v, x2f, x3f)
            if (i != ie+1) {
              uint64_t s = hasher(xv1) ^ (hasher(xf2)*2654435761ULL)
                         ^ (hasher(xf3)*40503ULL) ^ 1ULL;
              std::mt19937 gen(s);
              std::normal_distribution<> dist(0.0, noise_amp);
              a1(k,j,i) = dist(gen);
            }
            // A2 at (x1f, x2v, x3f)
            if (j != je+1) {
              uint64_t s = hasher(xf1) ^ (hasher(xv2)*2654435761ULL)
                         ^ (hasher(xf3)*40503ULL) ^ 2ULL;
              std::mt19937 gen(s);
              std::normal_distribution<> dist(0.0, noise_amp);
              a2(k,j,i) = dist(gen);
            }
            // A3 at (x1f, x2f, x3v)
            if (k != ke+1) {
              uint64_t s = hasher(xf1) ^ (hasher(xf2)*2654435761ULL)
                         ^ (hasher(xv3)*40503ULL) ^ 3ULL;
              std::mt19937 gen(s);
              std::normal_distribution<> dist(0.0, noise_amp);
              a3(k,j,i) = dist(gen);
            }
          }
        }
      }
    }
  }

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        x1 = pcoord->x1v(i);
        x3 = pcoord->x3v(k);

        if (ipert == 1) {
          rval = amp*(ran2(&iseed) - 0.5);
          rd = den*std::exp(-x3*x3/(Hbeta*Hbeta))*(1.0+2.0*rval);
          if (rd < dfloor) rd = dfloor;
          SumRd += rd;
          if (NON_BAROTROPIC_EOS) {
            rp = pres/den*rd;
            if (rp < pfloor) rp = pfloor;
          }
          rval = amp*(ran2(&iseed) - 0.5);
          rvx = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvx += rd*rvx;

          rval = amp*(ran2(&iseed) - 0.5);
          rvy = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvy += rd*rvy;

          rval = amp*(ran2(&iseed) - 0.5);
          rvz = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvz += rd*rvz;
        } else {
          rd = den*std::exp(-x3*x3/(Hbeta*Hbeta));
          rvx = 0;
          rvy = 0;
          rvz = 0;
        }

        if (iv == 1) {
          rvx += 0.01 * (psi_f/0.01) * iso_cs * std::sin(2*PI*x3/Lz);
        }

        phydro->u(IDN,k,j,i) = rd;
        phydro->u(IM1,k,j,i) = rd*rvx;
        phydro->u(IM2,k,j,i) = rd*rvy;
        if(!porb->orbital_advection_defined)
          phydro->u(IM2,k,j,i) -= rd*qshear*Omega_0*x1;
        phydro->u(IM3,k,j,i) = rd*rvz;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = rp/(gam-1.0)
                                 + 0.5*(SQR(phydro->u(IM1,k,j,i))
                                        + SQR(phydro->u(IM2,k,j,i))
                                        + SQR(phydro->u(IM3,k,j,i)))/rd;
        }

        if (MAGNETIC_FIELDS_ENABLED) {
          if (ifield == 1) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
          }
          if (ifield == 2) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = B0;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0;
          }
          if (ifield == 3) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = B0*(1.0+0.5*std::sin(static_cast<Real>(kx)*x1));
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0*(1.0 + 0.5*
                                                     std::sin(static_cast<Real>(kx)*x1));
          }
          if (ifield == 4) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = B0*(std::cos(static_cast<Real>(kx)*x1));
            pfield->b.x3f(k,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = B0*(std::cos(static_cast<Real>(kx)*x1));
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
          }
          if (ifield == 5 && std::abs(x3) < 2.0) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = B0;
            pfield->b.x3f(k,j,i) = 0.0;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = B0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = 0.0;
          }
          if (ifield == 6) {
            // net toroidal field with constant beta with height
            Real By_local = std::sqrt(2.0*pres/beta
                            * std::exp(-x3*x3/(Hbeta*Hbeta)));
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = By_local;
            pfield->b.x3f(k,j,i) = 0.0;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = By_local;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = 0.0;
          }
          if (ifield == 7) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = 0.0;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = 0.0;
          }
          if (ifield == 8) {
            pfield->b.x1f(k,j,i) = (a3(k  ,j+1,i) - a3(k,j,i))/pcoord->dx2f(j) -
                                   (a2(k+1,j  ,i) - a2(k,j,i))/pcoord->dx3f(k);
            pfield->b.x2f(k,j,i) = (a1(k+1,j,i  ) - a1(k,j,i))/pcoord->dx3f(k) -
                                   (a3(k  ,j,i+1) - a3(k,j,i))/pcoord->dx1f(i);
            pfield->b.x3f(k,j,i) = (a2(k,j  ,i+1) - a2(k,j,i))/pcoord->dx1f(i) -
                                   (a1(k,j+1,i  ) - a1(k,j,i))/pcoord->dx2f(j);
            if (i==ie) pfield->b.x1f(k,j,ie+1) =
                (a3(k,j+1,ie+1) - a3(k,j,ie+1))/pcoord->dx2f(j) -
                (a2(k+1,j,ie+1) - a2(k,j,ie+1))/pcoord->dx3f(k);
            if (j==je) pfield->b.x2f(k,je+1,i) =
                (a1(k+1,je+1,i) - a1(k,je+1,i))/pcoord->dx3f(k) -
                (a3(k,je+1,i+1) - a3(k,je+1,i))/pcoord->dx1f(i);
            if (k==ke) pfield->b.x3f(ke+1,j,i) =
                (a2(ke+1,j,i+1) - a2(ke+1,j,i))/pcoord->dx1f(i) -
                (a1(ke+1,j+1,i) - a1(ke+1,j,i))/pcoord->dx2f(j);
          }
        } // MHD
      }
    }
  }

  // For random perturbations as in HGB, ensure net momentum is zero by
  // subtracting off mean of perturbations
  if (ipert == 1) {
    if (lid == pmy_mesh->nblocal - 1) {
#ifdef MPI_PARALLEL
      MPI_Allreduce(MPI_IN_PLACE, &SumRd,  1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &SumRvx, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &SumRvy, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &SumRvz, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
      std::int64_t cell_num = pmy_mesh->GetTotalCells();
      SumRvx /= SumRd*cell_num;
      SumRvy /= SumRd*cell_num;
      SumRvz /= SumRd*cell_num;
      for (int b = 0; b < pmy_mesh->nblocal; ++b) {
        Hydro *ph = pmy_mesh->my_blocks(b)->phydro;
        for (int k=ks; k<=ke; k++) {
          for (int j=js; j<=je; j++) {
            for (int i=is; i<=ie; i++) {
              ph->u(IM1,k,j,i) -= ph->u(IDN,k,j,i)*SumRvx;
              ph->u(IM2,k,j,i) -= ph->u(IDN,k,j,i)*SumRvy;
              ph->u(IM3,k,j,i) -= ph->u(IDN,k,j,i)*SumRvz;
            }
          }
        }
      }
    }
  }
  return;
}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  return;
}

void MeshBlock::UserWorkInLoop() {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real& u_d  = phydro->u(IDN,k,j,i);
        u_d = (u_d > dfloor) ?  u_d : dfloor;
        if (NON_BAROTROPIC_EOS) {
          Real gam = peos->GetGamma();
          Real& w_p  = phydro->w(IPR,k,j,i);
          Real& u_e  = phydro->u(IEN,k,j,i);
          const Real& u_m1 = phydro->u(IM1,k,j,i);
          const Real& u_m2 = phydro->u(IM2,k,j,i);
          const Real& u_m3 = phydro->u(IM3,k,j,i);
          w_p = (w_p > pfloor) ?  w_p : pfloor;
          Real di = 1.0/u_d;
          Real ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
          u_e = w_p/(gam-1.0)+ke;
        }
      }
    }
  }
  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  return;
}


void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  Real fsmooth, xi, sign;
  Real Lz = pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min;
  Real z0 = Lz/2.0;
  Real lambda = 0.1 / z0;
  Real fw_x;
  Real rise_envelope = (rise_t > 0.0) ? (1.0 - std::exp(-time/rise_t)) : 1.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real den = prim(IDN,k,j,i);
        Real x3 = pmb->pcoord->x3v(k);

        // Vertical gravity with smoothing near boundaries
        if (std::abs(x3) > 1e-15) {
          if (x3 >= 0) {
            sign = -1.0;
          } else {
            sign = 1.0;
          }
          xi = z0/x3;
          fsmooth = SQR( std::sqrt( SQR(xi+sign) + SQR(xi*lambda) ) + xi*sign );
          cons(IM3,k,j,i) -= dt*den*SQR(Omega_0)*x3*fsmooth;
          if (NON_BAROTROPIC_EOS) {
            cons(IEN,k,j,i) -= dt*den*SQR(Omega_0)*prim(IVZ,k,j,i)*x3*fsmooth;
          }
        }

        // Sloshing forcing
        fw_x = den * SQR(Omega_0)*Hbeta*psi_f*rise_envelope
               *std::sin(2*PI*x3/Lz)*std::cos(Omega_0*time);
        cons(IM1,k,j,i) += dt*fw_x;
      }
    }
  }
  return;
}

//  Here is the lower z outflow boundary.
//  The basic idea is that the pressure and density
//  are exponentially extrapolated in the ghost zones
//  assuming a constant temperature there (i.e., an
//  isothermal atmosphere). The z velocity (NOT the
//  momentum) are set to zero in the ghost zones in the
//  case of the last lower physical zone having an inward
//  flow.  All other variables are extrapolated into the
//  ghost zones with zero slope.

void StratOutflowInnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim, FaceField &b,
                         Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          b.x1f(kl-k,j,i) = b.x1f(kl,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          b.x2f(kl-k,j,i) = b.x2f(kl,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(kl-k,j,i) = b.x3f(kl,j,i);
        }
      }
    }
  } // MHD

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(kl-k);
        Real x3b = pco->x3v(kl);
        Real den = prim(IDN,kl,j,i);
        Real Tkl = 0.5*SQR(Omega_0);
        if (NON_BAROTROPIC_EOS) {
          Real presskl = prim(IPR,kl,j,i);
          presskl = std::max(presskl,pfloor);
          Tkl = presskl/den;
        }
        // Extrapolate density using the magnetic scale height
        prim(IDN,kl-k,j,i) = den*std::exp(-(SQR(x3)-SQR(x3b))/
                                          (Hbeta*Hbeta));
        prim(IVX,kl-k,j,i) = prim(IVX,kl,j,i);
        prim(IVY,kl-k,j,i) = prim(IVY,kl,j,i);
        if (prim(IVZ,kl,j,i) >= 0.0) {
          prim(IVZ,kl-k,j,i) = 0.0;
        } else {
          prim(IVZ,kl-k,j,i) = prim(IVZ,kl,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,kl-k,j,i) = prim(IDN,kl-k,j,i)*Tkl;
      }
    }
  }
  return;
}

// Here is the upper z outflow boundary.
void StratOutflowOuterX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          b.x1f(ku+k,j,i) = b.x1f(ku,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          b.x2f(ku+k,j,i) = b.x2f(ku,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(ku+1+k,j,i) = b.x3f(ku+1,j,i);
        }
      }
    }
  } // MHD

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(ku+k);
        Real x3b = pco->x3v(ku);
        Real den = prim(IDN,ku,j,i);
        Real Tku = 0.5*SQR(Omega_0);
        if (NON_BAROTROPIC_EOS) {
          Real pressku = prim(IPR,ku,j,i);
          pressku = std::max(pressku,pfloor);
          Tku = pressku/den;
        }
        // Extrapolate density using the magnetic scale height
        prim(IDN,ku+k,j,i) = den*std::exp(-(SQR(x3)-SQR(x3b))/
                                          (Hbeta*Hbeta));
        prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
        prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
        if (prim(IVZ,ku,j,i) <= 0.0) {
          prim(IVZ,ku+k,j,i) = 0.0;
        } else {
          prim(IVZ,ku+k,j,i) = prim(IVZ,ku,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,ku+k,j,i) = prim(IDN,ku+k,j,i)*Tku;
      }
    }
  }
  return;
}

namespace {

Real HistoryBxBy(MeshBlock *pmb, int iout) {
  Real bxby = 0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &b = pmb->pfield->bcc;
  AthenaArray<Real> volume;
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        bxby -= volume(i)*b(IB1,k,j,i)*b(IB2,k,j,i);
      }
    }
  }
  return bxby;
}

Real HistorydVxVy(MeshBlock *pmb, int iout) {
  Real dvxvy = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  Real vshear = 0.0;
  AthenaArray<Real> volume;
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        if(!pmb->porb->orbital_advection_defined) {
          vshear = -qshear*Omega_0*pmb->pcoord->x1v(i);
        } else {
          vshear = 0.0;
        }
        dvxvy += volume(i)*w(IDN,k,j,i)*w(IVX,k,j,i)*(w(IVY,k,j,i) + vshear);
      }
    }
  }
  return dvxvy;
}
} // namespace
