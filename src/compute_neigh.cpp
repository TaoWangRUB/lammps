/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <mpi.h>
#include "compute_neigh.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "group.h"
#include "error.h"

#include "neighbor.h"
#include "neigh_list.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include <iostream>
#include <iomanip>
#include "pair.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeNeigh::ComputeNeigh(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 4) error->all(FLERR,"Illegal compute ke command");

  scalar_flag = 1;
  extscalar = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeNeigh::init()
{
  
}

/* ---------------------------------------------------------------------- */

double ComputeNeigh::compute_scalar()
{
  
  int i,j,k,ii,jj,kk,inum,jnum,jnumm1;
  int itype,jtype,ktype,ijparam,ikparam,ijkparam;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,rsq1,rsq2;
  double delr1[3],delr2[3],fj[3],fk[3];
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  Pair *pair = force->pair;
  double **cutsq = force->pair->cutsq;

   // EDIT
  // write neighbour lists every 100 steps
  if ( !(myStep % 10) ) {

    // decide number of samples for each time step
    //int chosenAtoms[] = {1, 6};
    int chosenAtoms[] = {307, 309};
    //for (int ii : chosenAtoms) {
    for (int ii=0; ii < inum; ii++) {
      i = ilist[ii];
      itype = type[i];

      // calculate energies manually, not eatom[i]
      double energy = 0;

      if (itype == 0) {
        outfiles[0].open(filename0.c_str(), std::ios::app);
      }
      else { 
        outfiles[1].open(filename1.c_str(), std::ios::app);
      }

      double xi = x[i][0];
      double yi = x[i][1];
      double zi = x[i][2];

      // write out coordinates of chosen atoms
      /*if (myStep == 0) {
        std::cout << "Chosen atom: "
        << i << " " << itype << " " << xi << " " << yi << " " 
        << zi << " " << std::endl;  
      }*/

      jlist = firstneigh[i];
      jnum = numneigh[i];
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        jtag = tag[j];
        jtype = type[j];

        delr1[0] = x[j][0] - xi;
        delr1[1] = x[j][1] - yi;
        delr1[2] = x[j][2] - zi;

        rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
  
        // pair cut
        if (rsq1 >= cutsq[itype][jtype]) continue;

        //twobody(&params[ijparam],rsq1,fpair,eflag,evdwl);
        evdwl = 1.0;
        energy += evdwl/2;

        // write relative coordinates to file
        // check triplet cuts when making symmetry later
        outfiles[itype] << std::setprecision(17) << delr1[0] << " " 
        << delr1[1] << " " << delr1[2] << " " << rsq1 << " " << jtype << " ";

        // triplet cut
        if (rsq1 >= cutsq[itype][jtype]) continue;

        for (kk = jj+1; kk < jnum; kk++) {
          k = jlist[kk];
          ktype = type[k];

          delr2[0] = x[k][0] - xi;
          delr2[1] = x[k][1] - yi;
          delr2[2] = x[k][2] - zi;
          rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];

          if (rsq2 >= cutsq[itype][jtype]) continue;

          //threebody(&params[ijparam],&params[ikparam],&params[ijkparam],
          //          rsq1,rsq2,delr1,delr2,fj,fk,eflag,evdwl);
          energy += evdwl;
        }
      }

      // store energy
      outfiles[itype] << std::setprecision(17) << energy << std::endl;
      outfiles[itype].close();
    }   
  }
  myStep++;

  scalar = 1;
  return scalar;
}
