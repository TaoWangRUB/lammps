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

#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "comm.h"
#include "memory.h"
#include <iostream>
#include <iomanip>
#include "pair.h"
#include <vector>

//#define EXTENDED_ERROR_CHECKING

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

  maxDelay = atoi(arg[3+nTypes]);
  useAlgo = atoi(arg[3+nTypes+1]);

  fprintf(screen, "\nuseAlgo: %d\n", useAlgo);
  fprintf(logfile, "\nuseAlgo: %d\n", useAlgo);

  scalar_flag = 1;
  extscalar = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeNeigh::init()
{
  chosenAtoms = force->pair->chosenAtoms;
  nChosenAtoms = chosenAtoms.size();
  
  fprintf(screen, "\nN chosen atoms in neigh: %d\n", nChosenAtoms);

  tau.resize(nChosenAtoms);
  for (int t=0; t < nChosenAtoms; t++) tau[t] = 1;

  sample.resize(nChosenAtoms);
  for (int s=0; s < nChosenAtoms; s++) sample[s] = 1;

  if (nTypes > 1) {
    filename0 = "Data/TrainingData/neighbours0.txt";
    filename1 = "Data/TrainingData/neighbours1.txt";
  }
  else
    filename0 = "Data/TrainingData/neighbours.txt";


  // create filenames
  char buffer[20];
  int buffer_length = 0;
  
  fprintf(screen, "\nN atom types: %d\n", nTypes);
  
  for (std::vector<int>::size_type i = 0; i != alpha.size(); i++){
    fprintf(screen, "\nalpha[%d] = %f\n", i, alpha[i]);
    buffer_length += sprintf(buffer + buffer_length, "%1.1f", alpha[i]);
    if (i == alpha.size() - 1){
        buffer_length += sprintf(buffer + buffer_length, ".txt");
        break;
    }
    else
        buffer_length += sprintf(buffer + buffer_length, "-");
        
  }
  std::string name(buffer);

  filenameTau = "1e4tau" + name;
  filenameStep = "1e4step" + name;
  fprintf(screen, "file been created: %s\n", filenameTau.c_str());
  fprintf(screen, "file been created: %s\n", filenameStep.c_str());

  // trying to open files, check if file successfully opened
  outfiles[0].open(filename0.c_str());
  if ( !outfiles[0].is_open() ) 
    std::cout << "File is not opened" << std::endl;
  outfiles[0].close();

  if (nTypes > 1) {
    outfiles[1].open(filename1.c_str());
    if ( !outfiles[1].is_open() ) 
      std::cout << "File is not opened" << std::endl;
    outfiles[1].close();
  }

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
  neighbor->requests[irequest]->pair = 0;       //
  neighbor->requests[irequest]->compute = 1;    // compute class require neighbor list
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;       // full neighbor list is required
  neighbor->requests[irequest]->occasional = 1; // neighbor list only updated when required
}

void ComputeNeigh::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

double ComputeNeigh::compute_scalar()
{
  int i,j,k,ii,jj,kk,inum, gnum,jnum,jnumm1;
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

  int *mask = atom->mask;

  inum = list->inum;                //# of I atoms neighbors are stored for
  gnum = list->gnum;                //# of ghost atoms neighbors are stored for
  ilist = list->ilist;              //local indices of neighbors for each I atom

  numneigh = list->numneigh;        //# of neighbors for each I atom
  firstneigh = list->firstneigh;    //ptr to 1st nearest neighbor list for each I atom

  //comm->reverse_comm_compute(this);
  //comm->forward_comm_compute(this);

  Pair *pair = force->pair;
  double **cutsq = force->pair->cutsq;
  std::vector<double> dumpEnergies = force->pair->dumpEnergies;

  int atomCount = 0;
  outStep.open(filenameStep, std::ios::app);
  for (auto i : chosenAtoms) {
  //for (i=0; i < inum; i++) {
    //if (!(mask[i] & groupbit)) continue;
    itype = type[i]-1;

    double F = sqrt(f[i][0]*f[i][0] + f[i][1]*f[i][1] + f[i][2]*f[i][2]);

    // update tau every 10th step
    if ( !(myStep % maxDelay) && myStep > 0) {

      // no delay if force is larger than alpha
      if (F > alpha[itype]) tau[atomCount] = 1;

      // calculate delay if force less than alpha
      else { 
        int factor = floor( alpha[itype] / F );
        if (factor > maxDelay) tau[atomCount] = maxDelay;
        else                   tau[atomCount] = factor;
      }
    }

    if (myStep > 0)
      if ( (myStep % tau[atomCount] > 0) && useAlgo) continue;

    outStep << tag[i] << " " << myStep << endl;

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

      if (rsq1 >= cutsq[itype+1][jtype+1]) continue;

      // write relative coordinates to file
      // check triplet cuts when making symmetry later

      if (nTypes > 1)
        outfiles[itype] << std::setprecision(17) << delr1[0] << " " 
        << delr1[1] << " " << delr1[2] << " " << rsq1 << " " << jtype << " ";
      else
        outfiles[itype] << std::setprecision(17) << delr1[0] << " " 
        << delr1[1] << " " << delr1[2] << " " << rsq1 << " ";
    }

    // store energy
    //cout << std::setprecision(10) << tag[i] << " " << dumpEnergies[atomCount] << " " << pair->eatom[i] << endl;
    outfiles[itype] << std::setprecision(17) << dumpEnergies[atomCount] << std::endl;
    outfiles[itype].close();
    atomCount++;
  }   
  outStep.close();

  // write tau and step sampled to file
  outTau.open(filenameTau, std::ios::app);
  for (int t : tau) outTau << t << " ";
  outTau << endl;
  outTau.close();
  myStep++;

  return 1;
}

