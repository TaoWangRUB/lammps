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

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "compute_pair_local.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "group.h"
#include "memory.h"
#include "error.h"

#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "comm.h"
#include "memory.h"
#include <iostream>
#include <iomanip>
#include "pair.h"
#include "pair_vashishta.h"
#include <vector>

using namespace LAMMPS_NS;

using std::cout;
using std::endl;

/* ---------------------------------------------------------------------- */

ComputeNeigh::ComputeNeigh(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 6 && narg != 7) error->all(FLERR,"Illegal compute neigh command");

  nTypes = atom->ntypes;
  alpha.resize(nTypes);
  for (int i=0; i < nTypes; i++)
    alpha[i] = atof(arg[3+i]);

  maxFactor = atoi(arg[3+nTypes]);
  useAlgo = atoi(arg[3+nTypes+1]);

  cout << "useAlgo: " << useAlgo << endl;

  scalar_flag = 1;
  extscalar = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeNeigh::init()
{
  chosenAtoms = force->pair->chosenAtoms;
  nChosenAtoms = chosenAtoms.size();

  tau.resize(nChosenAtoms);
  for (int t : tau) t = 0;

  sample.resize(nChosenAtoms);
  for (int s : sample) s = 0;

  filename0 = "Data/TrainingData/neighbours0.txt";
  filename1 = "Data/TrainingData/neighbours1.txt";

  // create filenames
  char buffer[20];
  cout << "nTypes: " << nTypes << endl;
  if (nTypes == 1)
    sprintf(buffer, "%1.1f.txt", alpha[0]);
  else
    sprintf(buffer, "%1.1f-%1.1f.txt", alpha[0], alpha[1]);
  std::string name(buffer);
  filenameTau = "tmp/1e4tau" + name;
  filenameStep = "tmp/1e4step" + name;
  cout << filenameTau << endl;
  cout << filenameStep << endl;

  // trying to open files, check if file successfully opened
  outfiles[0].open(filename0.c_str());
  if ( !outfiles[0].is_open() ) 
    std::cout << "File is not opened" << std::endl;
  outfiles[0].close();

  outfiles[1].open(filename1.c_str());
  if ( !outfiles[1].is_open() ) 
    std::cout << "File is not opened" << std::endl;
  outfiles[1].close();

  outTau.open(filenameTau);
  if ( !outTau.is_open() ) 
    std::cout << "File is not opened" << std::endl;
  outTau.close();

  outStep.open(filenameStep);
  if ( !outStep.is_open() ) 
    std::cout << "File is not opened" << std::endl;
  outStep.close();

  // request correct neighbour lists
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->occasional = 1;
}

void ComputeNeigh::init_list(int id, NeighList *ptr)
{
  list = ptr;
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
  double *ptr;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  Pair *pair = force->pair;
  double **cutsq = force->pair->cutsq;
  std::vector<double> dumpEnergies = force->pair->dumpEnergies;

  // write tau and step sampled to file
  outTau.open(filenameTau, std::ios::app);
  for (int t : tau) outTau << t << " ";
  outTau << endl;
  outTau.close();

  outStep.open(filenameStep, std::ios::app);
  for (int i=0; i < nChosenAtoms; i++)
    if (tau[i] == 0) outStep << i << " " << myStep << endl;
  outStep.close();

  // sample if tau = 0
  int counter = 0;
  for (ii : chosenAtoms) {
    i = ilist[ii];
    itype = type[i]-1;

    if (tau[counter] > 0 && useAlgo) {
      sample[counter] = 0;
      continue;
    }
    else sample[counter] = 1;

    double F = sqrt(f[i][0]*f[i][0] + f[i][1]*f[i][1] + f[i][2]*f[i][2]);

    // no delay if force is larger than alpha
    if (F > alpha[itype]) tau[counter] = 0;

    // calculate delay if force less than alpha
    else {
      int factor = floor( alpha[itype] / F );
      if (factor > maxFactor) factor = maxFactor;
      tau[counter] = factor;
    }

    if (itype == 0) outfiles[0].open(filename0.c_str(), std::ios::app);
    else            outfiles[1].open(filename1.c_str(), std::ios::app);

    double xi = x[i][0];
    double yi = x[i][1];
    double zi = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtag = tag[j];
      jtype = type[j]-1;

      delr1[0] = x[j][0] - xi;
      delr1[1] = x[j][1] - yi;
      delr1[2] = x[j][2] - zi;

      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];

      // pair cut
      if (cutsq[itype+1][jtype+1] != 30.25) {
        cout << "Wrong cut" << endl;
        exit(1);
      }

      if (rsq1 >= cutsq[itype+1][jtype+1]) continue;

      // write relative coordinates to file
      // check triplet cuts when making symmetry later
      outfiles[itype] << std::setprecision(17) << delr1[0] << " " 
      << delr1[1] << " " << delr1[2] << " " << rsq1 << " " << jtype << " ";
    }

    // store energy
    outfiles[itype] << std::setprecision(17) << dumpEnergies[counter] << std::endl;
    outfiles[itype].close();
    counter++;
  }   

  // adjust sampling time interval
  for (int i=0; i < nChosenAtoms; i++)
    if (tau[i] > 0 && sample[i] == 0) tau[i]--;

  myStep++;

  return 1;
}

