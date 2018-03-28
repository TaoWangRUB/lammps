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

// for parallel: the average energy & virial and center of mass velocity
// for the Evans force should be local. All other quantities should be global

#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "string.h"
#include "fix_evans_thermal.h"
#include "atom.h"
#include "compute.h"
#include "domain.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "group.h"
#include "update.h"
#include "pair.h"
#include "modify.h"
#include "comm.h"
#include "error.h"

//#define EXTENDED_ERROR_CHECKING
#ifdef EXTENDED_ERROR_CHECKING
#include <iostream>
#include <iomanip>
using std::cout;
using std::setprecision;
#endif
#include <iostream>
using std::cout;

using namespace LAMMPS_NS;
using namespace FixConst;

#define INVOKED_PERATOM 8

static const int vsize_ = 8; // size of output vector

/* ---------------------------------------------------------------------- */

FixEvansThermal::FixEvansThermal(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  useThermostat_(false), lambda_(0.0),
  step_(1),
  fieldX_(0.0), fieldY_(0.0), fieldZ_(0.0), M_(0.0), Mlocal_(0.0), Nlocal_(0),
  parallelPartition_(true)
{
  vBar_[0]=0; vBar_[1]=0; vBar_[2]=0;
  vBarLocal_[0]=0; vBarLocal_[1]=0; vBarLocal_[2]=0;
  J_[0]=0; J_[1]=0; J_[2]=0;
  // number of atoms in the group
  N_ = group->count(igroup);
  // compute total mass, once
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double m =  (mass) ? mass[type[i]] : rmass[i];
      M_ += m;
      Nlocal_++;
    }
  }
  Mlocal_ = M_;
  static const int ndata = 1;
  double local_data[ndata] ={M_};
  double data[ndata]={0};
  MPI_Allreduce(local_data,data,ndata,MPI_DOUBLE,MPI_SUM,world);
  M_ = data[0];
#ifdef EXTENDED_ERROR_CHECKING
  cout << "total mass " << M_ << "\n";
#endif

  // fix parses first 3 args, parse the rest here...
  for (int iarg = 3; iarg < narg; ++iarg) {
    if       (strcmp(arg[iarg],"field") == 0) {
      fieldX_ = atof(arg[++iarg]);
      fieldY_ = atof(arg[++iarg]);
      fieldZ_ = atof(arg[++iarg]);
    }
    else if  (strcmp(arg[iarg],"thermostat") == 0) {
      T0_ = atof(arg[++iarg]);
      noseFreq_ = 1.0/atof(arg[++iarg]);
      useThermostat_ = true;
    }
    // NOTE could delay this to init()
    else if (strcmp(arg[iarg],"pe") == 0)  {
      int idPE = modify->find_compute(arg[++iarg]); 
      if (idPE < 0) {
        error->all(FLERR,"fix evans: cannot find specified pe/atom compute");
      }
      if (modify->compute[idPE]->peatomflag == 0)
        error->all(FLERR,"fix evans: compute ID does not compute pe/atom");
      cPE_ =  modify->compute[idPE];
#ifdef EXTENDED_ERROR_CHECKING
      cout << "cPE_ " << cPE_ << " at creation\n";
#endif 
    }
    else if (strcmp(arg[iarg],"virial") == 0)  {
      int idVirial = modify->find_compute(arg[++iarg]);
      if (idVirial < 0) {
        error->all(FLERR,"fix evans: cannot find specified stress/atom compute");
      }
      if (modify->compute[idVirial]->pressatomflag == 0)
        error->all(FLERR,"fix evans: compute ID does not compute stress/atom");
      cVirial_ =  modify->compute[idVirial];
    }
    else { 
      error->all(FLERR,"Illegal fix evans command");
    }

  }
	// added lines
  grow_arrays(atom->nmax);
  
  // register callback to this fix from Atom class
  atom->add_callback(0); // 0 means has grow_arrays

  // output flags
  vector_flag = 1;
  size_vector = vsize_; // size of compute_vector
  extvector = 1;
  global_freq = 1;
}

/* ---------------------------------------------------------------------- */

FixEvansThermal::~FixEvansThermal() 
{
  // unregister callback to this fix from Atom class
  atom->delete_callback(id,0);
}

/* ---------------------------------------------------------------------- */

int FixEvansThermal::modify_param(int narg, char** arg)
{
  // parse arguments
  for (int iarg = 0; iarg < narg; ++iarg) {
    if        (strcmp(arg[iarg],"field") == 0) {
      fieldX_ = atof(arg[++iarg]);
      fieldY_ = atof(arg[++iarg]);
      fieldZ_ = atof(arg[++iarg]);
    }
    else { 
      error->all(FLERR,"Illegal fix_modify evans command");
    }
  }

  return narg; // return number of args parsed
}

/* ---------------------------------------------------------------------- */

int FixEvansThermal::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEvansThermal::init()
{
  // set timestep
  reset_dt();

  // update computes
#ifdef EXTENDED_ERROR_CHECKING
      cout << "cPE_ " << cPE_ << " at creation\n";
#endif 
  cPE_    ->addstep(update->ntimestep+1);
  cVirial_->addstep(update->ntimestep+1);

  // echo parameters
  int me = 0;
  MPI_Comm_rank(world,&me);
  if (me == 0) {
      fprintf(screen,"\n\n Fix_evans... \n\n");
      fprintf(screen,"  fieldX_ value is : %f \n", fieldX_);
	  fprintf(screen,"  fieldX_ value is : %f \n", fieldY_);
	  fprintf(screen,"  fieldX_ value is : %f \n", fieldZ_);
    if (useThermostat_) 
	  fprintf(screen,"  set temperature : %f \n", T0_);
  }
    
    // need a half neighbor list, built when ever re-neighboring occurs
    int irequest = neighbor->request((void *) this);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->fix = 1;
    neighbor->requests[irequest]->half = 1; // i.e. newton_pair is true
    neighbor->requests[irequest]->full = 0;
}

/*----------------------------------------------------------------------- */
void FixEvansThermal::initial_integrate(int vflag)
{
  double dtfm;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double dthalf = 0.5*(update->dt);
  double *special_coul = force->special_coul;       
  double *special_lj = force->special_lj;
  double **cutsq = force->pair->cutsq; // sq. of cutoff radius    
  int nall = atom->nlocal + atom->nghost;
  // initializing r_ij average for C_F 
  int n = atom->nlocal;
  if (force->newton_pair) n += atom->nghost;
    
  double t_current = compute_thermo(); // stores T_                                     //A1
  // thermostat
  if (useThermostat_) {
      // compute T
      // fwd euler:  \dot{s} = 1/m_s (sum_i p_i . p_i / m_i - 3N kB T)
      lambda_ += 0.5 * dthalf * noseFreq_ * noseFreq_ * (t_current / T0_ - 1.0);        //A2
      // exp update
      double v_scale = exp(-1.0 * dthalf*lambda_);                                      //A3
      for (int i = 0; i < nlocal; i++) {
          if (mask[i] & groupbit) {
              v[i][0] *= v_scale;
              v[i][1] *= v_scale;
              v[i][2] *= v_scale;
          }
      }
      t_current *= exp(-2.0 * dthalf * lambda_);                                         //A4
      T_ = t_current;
      lambda_ += 0.5 * dthalf * noseFreq_ * noseFreq_ * (t_current / T0_ - 1.0);        //A5
  }

  // first Verlet (half) step, v_n -> v_n+1/2, x_n -> x_n+1/2
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double  m =  (mass) ? mass[type[i]] : rmass[i];
      dtfm = dtf / m;
      v[i][0] += dtfm * f[i][0];                                                        //A6
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];
      x[i][0] += dthalf * v[i][0];                                                      //A7
      x[i][1] += dthalf * v[i][1];
      x[i][2] += dthalf * v[i][2];
    }
  }
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd;
    double xbox = domain->boxhi[0] - domain->boxlo[0];
    double ybox = domain->boxhi[1] - domain->boxlo[1];
    double zbox = domain->boxhi[2] - domain->boxlo[2];
    
    int inum = 0, jnum = 0, i = 0, j = 0, itype, jtype;
    int *ilist = NULL, *jlist = NULL, *numneigh = NULL, **firstneigh = NULL;
    double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
    inum = list_->inum;
    ilist = list_->ilist;
    numneigh = list_->numneigh;
    firstneigh = list_->firstneigh;
    for (int ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        itype = type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];
        
        int icount = 0;
        double sumx[3] = {0.0, 0.0, 0.0};
        for (int jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;
            
            /*delx = xtmp - (x[j][0] + ((int)((xtmp - x[j][0]) / xbox * 2.0)) * xbox);
            dely = ytmp - (x[j][1] + ((int)((xtmp - x[j][1]) / ybox * 2.0)) * ybox);
            delz = ztmp - (x[j][2] + ((int)((xtmp - x[j][2]) / zbox * 2.0)) * zbox);
            rsq = delx*delx + dely*dely + delz*delz;
            jtype = type[j];*/
            //if (rsq < cutsq[itype][jtype]) {
            sumx[0] += x[j][0]; // + ((int)((xtmp - x[j][0]) / xbox * 2.0)) * xbox;
            sumx[1] += x[j][1]; // + ((int)((xtmp - x[j][1]) / ybox * 2.0)) * ybox;
            sumx[2] += x[j][2]; // + ((int)((xtmp - x[j][2]) / zbox * 2.0)) * zbox;
            icount++;
            //}
        }
        
        double xrcm[3] = {0.0, 0.0, 0.0};
        if (icount != 0) {
            xrcm[0] = x[i][0] - sumx[0] / icount;
            xrcm[1] = x[i][1] - sumx[1] / icount;
            xrcm[2] = x[i][2] - sumx[2] / icount;
        }
        double CFx = 0.0 - (xrcm[0]*v[i][0]*fieldX_ + xrcm[0]*v[i][1]*fieldY_ + xrcm[0]*v[i][2]*fieldZ_);
        double CFy = 0.0 - (xrcm[1]*v[i][0]*fieldX_ + xrcm[1]*v[i][1]*fieldY_ + xrcm[1]*v[i][2]*fieldZ_);
        double CFz = 0.0 - (xrcm[2]*v[i][0]*fieldX_ + xrcm[2]*v[i][1]*fieldY_ + xrcm[2]*v[i][2]*fieldZ_);
        x[i][0] += dtv * CFx;                                                           //A8
        x[i][1] += dtv * CFy;
        x[i][2] += dtv * CFz;
        x[i][0] += dthalf * v[i][0];                                                    //A9
        x[i][1] += dthalf * v[i][1];
        x[i][2] += dthalf * v[i][2];
    }

}

/* ---------------------------------------------------------------------- */
/*    force is updated at this moment */
void FixEvansThermal::final_integrate()
{
  double dtfm;
  double dt = update->dt;
  double **v = atom->v;
  double **f = atom->f;
  double **x = atom->x;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double half_mvv2e = 0.5*(force->mvv2e); // conversion factor
  double nktv2p = -force->nktv2p;
  double vscale = 1.0/nktv2p;
  double dthalf = 0.5*(update->dt);

#ifdef EXTENDED_ERROR_CHECKING
//  cout << "cPE_ " << cPE_ << " in final_integrate\n";
#endif 
  // invoke computes
  if (!(cPE_->invoked_flag & INVOKED_PERATOM)) {
    cPE_->compute_peratom();
    cPE_->invoked_flag |= INVOKED_PERATOM;
  }
  if (!(cVirial_->invoked_flag & INVOKED_PERATOM)) {
    cVirial_->compute_peratom();
    cVirial_->invoked_flag |= INVOKED_PERATOM;
  }
  double *pe      = cPE_->vector_atom;
  double **virial = cVirial_->array_atom;

  // add control force at (x_n+1, f_n+1) to total force
  aveDFx_ = aveDFy_ = aveDFz_ = 0.0;
    double sumFr = 0.0;
    int inum = 0, jnum = 0, i = 0, j = 0, itype, jtype;
    int *ilist = NULL, *jlist = NULL, *numneigh = NULL, **firstneigh = NULL;
    double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
    double **cutsq = force->pair->cutsq; // sq. of cutoff radius
    
    inum = list_->inum;
    ilist = list_->ilist;
    numneigh = list_->numneigh;
    firstneigh = list_->firstneigh;
    
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd;
    double xbox = domain->boxhi[0] - domain->boxlo[0];
    double ybox = domain->boxhi[1] - domain->boxlo[1];
    double zbox = domain->boxhi[2] - domain->boxlo[2];
    
    for (int ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        itype = type[i];
        if (mask[i] & groupbit) {
            xtmp = x[i][0];
            ytmp = x[i][1];
            ztmp = x[i][2];
            double  mi =  (mass) ? mass[type[i]] : rmass[i];
            double* vi = v[i];
            double* fi = f[i];                                                          //A10
            double ke =  half_mvv2e * mi * (vi[0]*vi[0] + vi[1]*vi[1] + vi[2]*vi[2]);
            double eng = pe[i] + ke;
            
            double sxx = vscale*virial[i][0];
            double syy = vscale*virial[i][1];
            double szz = vscale*virial[i][2];
            double sxy = vscale*virial[i][3];
            double sxz = vscale*virial[i][4];
            double syz = vscale*virial[i][5];
            
            jlist = firstneigh[i];
            jnum = numneigh[i];
        
            int icount = 0;
            double sumx[3] = {0.0, 0.0, 0.0};
            for (int jj = 0; jj < jnum; jj++) {
                j = jlist[jj];
                j &= NEIGHMASK;
                
                /*delx = xtmp - (x[j][0] + ((int)((xtmp - x[j][0]) / xbox * 2.0)) * xbox);
                dely = ytmp - (x[j][1] + ((int)((xtmp - x[j][1]) / ybox * 2.0)) * ybox);
                delz = ztmp - (x[j][2] + ((int)((xtmp - x[j][2]) / zbox * 2.0)) * zbox);
                rsq = delx*delx + dely*dely + delz*delz;
                jtype = type[j];*/
                //if (rsq < cutsq[itype][jtype]) {
                    sumx[0] += x[j][0]; // + ((int)((xtmp - x[j][0]) / xbox * 2.0)) * xbox;
                    sumx[1] += x[j][1]; // + ((int)((xtmp - x[j][1]) / ybox * 2.0)) * ybox;
                    sumx[2] += x[j][2]; // + ((int)((xtmp - x[j][2]) / zbox * 2.0)) * zbox;
                    icount++;
                //}
            }
            
            double xrcm[3] = {0.0, 0.0, 0.0};
            if (icount != 0) {
                xrcm[0] = x[i][0] - sumx[0] / icount;
                xrcm[1] = x[i][1] - sumx[1] / icount;
                xrcm[2] = x[i][2] - sumx[2] / icount;
            }

            double syx = sxy;
            double szx = sxz;
            double szy = syz;
    
            double DFx = (eng + sxx)*fieldX_ + sxy*fieldY_ + sxz*fieldZ_;
            double DFy = syx*fieldX_ + (eng + syy)*fieldY_ + syz*fieldZ_;
            double DFz = szx*fieldX_ + szy*fieldY_ + (eng + szz)*fieldZ_;
            double factor = fi[0] * xrcm[0] + fi[1] * xrcm[1] + fi[2] * xrcm[2];
            DFx -= factor * fieldX_;
            DFy -= factor * fieldY_;
            DFz -= factor * fieldZ_;
            fi[0] += DFx;                                                               //A11
            fi[1] += DFy;
            fi[2] += DFz;
            aveDFx_ += DFx;
            aveDFy_ += DFy;
            aveDFz_ += DFz;
        }
    }

    // sum across processors
    static const int ndata = 3;
    double local_data[ndata] ={aveDFx_,aveDFy_,aveDFz_};
    double data[ndata] = {0,0,0};
    MPI_Allreduce(local_data,data,ndata,MPI_DOUBLE,MPI_SUM,world);
    aveDFx_    = data[0] / M_;
    aveDFy_    = data[1] / M_;
    aveDFz_    = data[2] / M_;
    

  // second  (half) Verlet step, v_n+1/2 -> v_n+1
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double  m =  (mass) ? mass[type[i]] : rmass[i];
      dtfm = dtf / m;
      double * fi = f[i];
      fi[0] -= aveDFx_ * m;
      fi[1] -= aveDFy_ * m;
      fi[2] -= aveDFz_ * m;
      v[i][0] += dtfm * fi[0];                                                          //A12
      v[i][1] += dtfm * fi[1];
      v[i][2] += dtfm * fi[2];
    }
  }
   
    // thermostat
    // compute T
    double t_current = compute_thermo(); // stores T_                                   //A13
    if (useThermostat_) {
        // fwd euler:  \dot{s} = 1/m_s (sum_i p_i . p_i / m_i - 3N kB T)
        lambda_ += 0.5 * dthalf * noseFreq_ * noseFreq_ * ( t_current / T0_ - 1.0);     //A14
        // exp update
        double v_scale = exp(-1.0 * dthalf*lambda_);                                    //A15
        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
                v[i][0] *= v_scale;
                v[i][1] *= v_scale;
                v[i][2] *= v_scale;
            }
        }
        t_current *= exp(-2.0 * dthalf * lambda_);                                       //A16
        T_ = t_current;
        lambda_ += 0.5 * dthalf * noseFreq_ * noseFreq_ * (t_current / T0_ - 1.0);       //A17
    }
    //cout << "velocity of atom 1 : " << v[1][0] << "\n";
    
    // request compute data for next step
    cPE_    ->addstep(update->ntimestep+1);
    cVirial_->addstep(update->ntimestep+1);
}

/* ---------------------------------------------------------------------- */
double FixEvansThermal::compute_thermo(void) 
{
  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double kB = force->boltz/force->mvv2e; // NOTE moving these changes output
  double dof = (int) domain->dimension * N_; // NOTE using 3N not 3N-3

  vBar_[0] = 0.0; vBar_[1] = 0.0; vBar_[2] = 0.0;
  T_ = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double m =  (mass) ? mass[type[i]] : rmass[i];
      vBar_[0] += m*v[i][0];
      vBar_[1] += m*v[i][1];
      vBar_[2] += m*v[i][2];
      T_ += m*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2]);
    }
  }
  // for "parallel_partition" of Evans forces
  vBarLocal_[0] = vBar_[0]/Mlocal_;
  vBarLocal_[1] = vBar_[1]/Mlocal_;
  vBarLocal_[2] = vBar_[2]/Mlocal_;
#ifdef EXTENDED_ERROR_CHECKING
  int me = 0;
  MPI_Comm_rank(world,&me);
  cout << "processor:"<<me<<" vbar = "<< vBarLocal_[0] << " " <<vBarLocal_[0] << " " << vBarLocal_[0] << "\n" << std::flush;
#endif

  // compute global quantities
  static const int ndata = 4; 
  double local_data[ndata]
    ={vBar_[0],vBar_[1],vBar_[2],T_};
  double data[ndata]={0,0,0,0};
  MPI_Allreduce(local_data,data,ndata,MPI_DOUBLE,MPI_SUM,world);
  vBar_[0] = data[0]/M_; vBar_[1] = data[1]/M_; vBar_[2] = data[2]/M_;
  return T_ = (data[3] -(vBar_[0]*vBar_[0] + vBar_[1]*vBar_[1] + vBar_[2]*vBar_[2]) / M_)/(dof*kB);
}


/* ---------------------------------------------------------------------- */

double FixEvansThermal::compute_vector(int n) 
{
#ifdef EXTENDED_ERROR_CHECKING
//  cout << "cPE_ " << cPE_ << " in compute_vector\n";
#endif 
  /*static const double tol = 1.0e-8;
    if (fabs(vBar_[0]*vBar_[0]+vBar_[0]*vBar_[0]+vBar_[0]*vBar_[0]) > tol) {
    cout << "WARNING: fix_evans_thermal, momentum is not being preserved\n" << std::flush;
    cout<<"aveV "<<vBar_[0]<<" "<<vBar_[1]<<" "<<vBar_[2]<<std::endl;
    }*/
  // compute J_Q
  if (n < 3) {
    double **v = atom->v;
    double *mass = atom->mass;
    double *rmass = atom->rmass;
    int *type = atom->type;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    double half_mvv2e = 0.5*(force->mvv2e); // conversion factor
    double *pe      = cPE_->vector_atom;
    double **virial = cVirial_->array_atom;


    J_[0] = 0; J_[1] = 0; J_[2] = 0;
    double Jv[3] = {0,0,0};
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        double m;
        if (mass)  m = mass[type[i]];
        else       m = rmass[i];
        double* vi = v[i];
        double ke =  half_mvv2e * m * (vi[0]*vi[0] + vi[1]*vi[1] + vi[2]*vi[2]);
        double e = pe[i]+ke;
        J_[0] += e*vi[0];
        J_[1] += e*vi[1];
        J_[2] += e*vi[2];
        Jv[0] -= virial[i][0]*vi[0] + virial[i][3]*vi[1] + virial[i][4]*vi[2];
        Jv[1] -= virial[i][3]*vi[0] + virial[i][1]*vi[1] + virial[i][5]*vi[2];
        Jv[2] -= virial[i][4]*vi[0] + virial[i][5]*vi[1] + virial[i][2]*vi[2];
      }
    }
    double nktv2p = force->nktv2p;
    double vscale = 1.0/nktv2p;

//cout << "Virial from compute_vector:  " << vscale*virial[1][0] << "\n";

    J_[0] += vscale*Jv[0];
    J_[1] += vscale*Jv[1];
    J_[2] += vscale*Jv[2];
    // sum heat flux across processors
    static const int ndata = 3;
    double local_data[ndata] ={J_[0],J_[1],J_[2]};
    double data[ndata] = {0,0,0};
    MPI_Allreduce(local_data,data,ndata,MPI_DOUBLE,MPI_SUM,world);
    J_[0] = data[0];
    J_[1] = data[1];
    J_[2] = data[2];
  }
  if      (n == 0) {
    return J_[0];
  } 
  else if (n == 1) {
    return J_[1];
  } 
  else if (n == 2) {
    return J_[2];
  }
  else if (n == 3) {
    return vBar_[0];
  } 
  else if (n == 4) {
    return vBar_[1];
  } 
  else if (n == 5) {
    return vBar_[2];
  } 
  else if (n == 6) { 
    return T_;
  }
  else if (n == 7) {
    return lambda_;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void FixEvansThermal::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dtq = 0.5 * update->dt;
}

void FixEvansThermal::init_list(int id, NeighList *ptr)
{
    list_ = ptr;
}


