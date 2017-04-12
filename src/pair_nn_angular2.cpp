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

/* ----------------------------------------------------------------------
   Contributing author:  Yongnan Xiong (HNU), xyn@hnu.edu.cn
                         Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_nn_angular2.h"
#include "atom.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include <fenv.h>

#include <fstream>
#include <iomanip>

using namespace LAMMPS_NS;
using std::cout;
using std::endl;

//#define MAXLINE 1024
//#define DELTA 4

/* ---------------------------------------------------------------------- */

PairNNAngular2::PairNNAngular2(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0; // We don't provide the force between two atoms only since it is Angular2
  restartinfo = 0;   // We don't write anything to restart file
  one_coeff = 1;     // only one coeff * * call
  manybody_flag = 1; // Not only a pair style since energies are computed from more than one neighbor
  cutoff = 10.0;      // Will be read from command line
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairNNAngular2::~PairNNAngular2()
{
  if (copymode) return;
  // If you allocate stuff you should delete and deallocate here. 
  // Allocation of normal vectors/matrices (not armadillo), should be created with
  // memory->create(...)

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

double PairNNAngular2::network(arma::mat inputVector) {
    // inputGraph vector is a 1xinputGraphs vector

    // linear activation for inputGraph layer
    m_preActivations[0] = inputVector;
    m_activations[0] = m_preActivations[0];

    // hidden layers
    for (int i=0; i < m_nLayers; i++) {
        // weights and biases starts at first hidden layer:
        // weights[0] are the weights connecting inputGraph layer to first hidden layer
        m_preActivations[i+1] = m_activations[i]*m_weights[i] + m_biases[i];
        m_activations[i+1] = sigmoid(m_preActivations[i+1]);
    }

    // linear activation for output layer
    m_preActivations[m_nLayers+1] = m_activations[m_nLayers]*m_weights[m_nLayers] + m_biases[m_nLayers];
    m_activations[m_nLayers+1] = m_preActivations[m_nLayers+1];

    // return activation of output neuron
    return m_activations[m_nLayers+1](0,0);
}

arma::mat PairNNAngular2::backPropagation() {
  // find derivate of output w.r.t. intput, i.e. dE/dr_ij
  // need to find the "error" terms for all the nodes in all the layers

  // the derivative of the output neuron's activation function w.r.t.
  // its inputGraph is propagated backwards.
  // the output activation function is f(x) = x, so this is 1
  arma::mat output(1,1); output.fill(1);
  m_derivatives[m_nLayers+1] = output;

  // we can thus compute the error vectors for the other layers
  for (int i=m_nLayers; i > 0; i--) {
      m_derivatives[i] = ( m_derivatives[i+1]*m_weightsTransposed[i] ) %
                         sigmoidDerivative(m_preActivations[i]);
  }

  // linear activation function for inputGraph neurons
  m_derivatives[0] = m_derivatives[1]*m_weightsTransposed[0];

  return m_derivatives[0];
}

arma::mat PairNNAngular2::sigmoid(arma::mat matrix) {

  return 1.0/(1 + arma::exp(-matrix));
}

arma::mat PairNNAngular2::sigmoidDerivative(arma::mat matrix) {

  arma::mat sigmoidMatrix = sigmoid(matrix);
  return sigmoidMatrix % (1 - sigmoidMatrix);
}

arma::mat PairNNAngular2::Fc(arma::mat R, double Rc, bool cut) {

  arma::mat value = 0.5*(arma::cos(m_pi*R/Rc) + 1);

  if (cut)
    for (int i=0; i < arma::size(R)(1); i++)
      if (R(0,i) > Rc) 
        value(0,i) = 0;
  
  return value;
}

double PairNNAngular2::Fc(double R, double Rc, bool cut) {

  if (cut)
    if (R < Rc)
      return 0.5*(cos(m_pi*R/Rc) + 1);
    else
      return 0;
  else
    return 0.5*(cos(m_pi*R/Rc) + 1);
}

arma::mat PairNNAngular2::dFcdR(arma::mat R, double Rc, bool cut) {

  double Rcinv = 1.0/Rc;
  arma::mat value = -(0.5*m_pi*Rcinv) * arma::sin(m_pi*R*Rcinv);

  if (cut)
    for (int i=0; i < arma::size(R)(1); i++)
      if (R(0,i) > Rc) 
        value(0,i) = 0;
  
  return value; 
}

double PairNNAngular2::dFcdR(double R, double Rc, bool cut) {

  double Rcinv = 1.0/Rc;

  if (cut)
    if (R < Rc)
      return -(0.5*m_pi*Rcinv) * sin(m_pi*R*Rcinv);
    else
      return 0;
  else
    return -(0.5*m_pi*Rcinv) * sin(m_pi*R*Rcinv);
}

double PairNNAngular2::G1(arma::mat Rij, double Rc) {

  return arma::accu( Fc(Rij, Rc, false) );
}

arma::mat PairNNAngular2::dG1dR(arma::mat Rij, double Rc) {

  return dFcdR(Rij, Rc, false);
}

double PairNNAngular2::G2(double rij, double eta, double Rc, double Rs) {

  return exp(-eta*(rij - Rs)*(rij - Rs)) * Fc(rij, Rc, false);
}

void PairNNAngular2::dG2dR(arma::mat Rij, double eta, double Rc, double Rs,
                          arma::mat &dG2) {

  dG2 = arma::exp(-eta*(Rij - Rs)%(Rij - Rs)) % 
          ( 2*eta*(Rs - Rij) % Fc(Rij, Rc, false) + dFcdR(Rij, Rc, false) );
}

double PairNNAngular2::G4(double rij, double rik, double rjk, double cosTheta, 
                          double eta, double Rc, double zeta, double lambda) {

  return pow(2, 1-zeta) * 
         pow(1 + lambda*cosTheta, zeta) *  
         exp( -eta*(rij*rij + rik*rik + rjk*rjk) ) * 
         Fc(rij, Rc, false) * Fc(rik, Rc, false) * Fc(rjk, Rc, true);
}

double PairNNAngular2::G5(double rij, double rik, double cosTheta, 
                          double eta, double Rc, double zeta, double lambda) {

  return pow(2, 1-zeta) * 
         pow(1 + lambda*cosTheta, zeta) *  
         exp( -eta*(rij*rij + rik*rik) ) * 
         Fc(rij, Rc, false) * Fc(rik, Rc, false);
}


void PairNNAngular2::dG4dR(double Rij, arma::mat Rik, arma::mat Rjk, 
                          arma::mat cosTheta, double eta, double Rc, 
                          double zeta, double lambda,
                          arma::mat &dEdRj3, arma::mat &dEdRk3,
                          arma::mat drij, arma::mat drik, arma::mat drjk) {

  arma::mat powCosThetaM1 = pow(2, 1-zeta)*arma::pow(1 + lambda*cosTheta, zeta-1);
  arma::mat F1 = powCosThetaM1 % (1 + lambda*cosTheta);

  arma::mat F2 = arma::exp(-eta*(Rij*Rij + Rik%Rik + Rjk%Rjk));

  double FcRij = Fc(Rij, Rc, false);
  arma::mat FcRik = Fc(Rik, Rc, false);
  arma::mat FcRjk = Fc(Rjk, Rc, true);
  arma::mat F3 = FcRij * FcRik % FcRjk;

  arma::mat K = lambda*zeta*powCosThetaM1;
  arma::mat L = -2*eta*F2;
  double Mij = dFcdR(Rij, Rc, false);
  arma::mat Mik = dFcdR(Rik, Rc, false);
  arma::mat Mjk = dFcdR(Rjk, Rc, true);

  arma::mat term1 = K % F2 % F3;
  arma::mat term2 = F1 % L % F3;

  arma::mat F1F2 = F1 % F2;
  arma::mat term3ij = F1F2 % FcRik % FcRjk * Mij;
  arma::mat term3ik = F1F2 % FcRjk % Mik * FcRij;

  arma::mat termjk = F1F2 % Mjk % FcRik * FcRij;

  double RijInv = 1.0 / Rij;
  arma::mat cosRijInv2 = cosTheta * RijInv*RijInv;

  arma::mat RikInv = 1.0 / Rik;
  arma::mat cosRikInv2 = cosTheta % RikInv%RikInv;

  arma::mat RijRikInv = RijInv * RikInv;

  arma::mat termij = (cosRijInv2 % term1) - term2 - RijInv*term3ij;
  arma::mat termik = (cosRikInv2 % term1) - term2 - RikInv%term3ik;
  arma::mat crossTerm = -term1 % RijRikInv; 
  arma::mat crossTermjk = termjk / Rjk;


  // all k's give a triplet energy contributon to atom j
  dEdRj3.row(0) =  (drij(0,0) * termij) + (drik.row(0) % crossTerm) -  
                   (drjk.row(0) % crossTermjk);
  dEdRj3.row(1) =  (drij(1,0) * termij) + (drik.row(1) % crossTerm) -  
                   (drjk.row(1) % crossTermjk);
  dEdRj3.row(2) =  (drij(2,0) * termij) + (drik.row(2) % crossTerm) -  
                   (drjk.row(2) % crossTermjk);

  // need all the different components of k's forces to do sum k > j
  dEdRk3.row(0) =  (drik.row(0) % termik) + (drij(0,0) * crossTerm) +  
                   (drjk.row(0) % crossTermjk);
  dEdRk3.row(1) =  (drik.row(1) % termik) + (drij(1,0) * crossTerm) +  
                   (drjk.row(1) % crossTermjk);
  dEdRk3.row(2) =  (drik.row(2) % termik) + (drij(2,0) * crossTerm) +  
                   (drjk.row(2) % crossTermjk);
}

void PairNNAngular2::dG4dj(double xj, double yj, double zj, 
               double xk, double yk, double zk, 
               double Rj, double Rk, double Rjk, double CosTheta,
               double eta, double Rc, double zeta, double Lambda, 
               double *dGj) {

  double Rj2 = Rj*Rj;
  double Rk2 = Rk*Rk;
  double Rjk2 = Rjk*Rjk;
  double Fcj = Fc(Rj, Rc, false);
  double Fck = Fc(Rk, Rc, false);
  double Fcjk = Fc(Rjk, Rc, true);
  double dFcj = dFcdR(Rj, Rc, false);
  double dFcjk = dFcdR(Rjk, Rc, true);

  dGj[0] = pow(2, -zeta)*Fck*(-2*Fcj*Fcjk*Lambda*Rjk*zeta*
    pow(CosTheta*Lambda + 1, zeta)*(CosTheta*Rk*xj - Rj*xk) - 
    4*Fcj*Fcjk*pow(Rj, 2)*Rjk*Rk*eta*(2*xj - xk)*
    pow(CosTheta*Lambda + 1, zeta + 1) + 2.0*Fcj*pow(Rj, 2)*
    Rk*dFcjk*(xj - xk)*pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Fcjk*Rj*Rjk*Rk*dFcj*xj*pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rjk2 + Rk2))/(pow(Rj, 2)*Rjk*Rk*
    (CosTheta*Lambda + 1));

  dGj[1] = pow(2, -zeta)*Fck*(-2*Fcj*Fcjk*Lambda*Rjk*zeta*
    pow(CosTheta*Lambda + 1, zeta)*(CosTheta*Rk*yj - Rj*yk) - 
    4*Fcj*Fcjk*pow(Rj, 2)*Rjk*Rk*eta*(2*yj - yk)*
    pow(CosTheta*Lambda + 1, zeta + 1) + 2.0*Fcj*pow(Rj, 2)*
    Rk*dFcjk*(yj - yk)*pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Fcjk*Rj*Rjk*Rk*dFcj*yj*pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rjk2 + Rk2))/(pow(Rj, 2)*Rjk*Rk*
    (CosTheta*Lambda + 1));

  dGj[2] = pow(2, -zeta)*Fck*(-2*Fcj*Fcjk*Lambda*Rjk*zeta*
    pow(CosTheta*Lambda + 1, zeta)*(CosTheta*Rk*zj - Rj*zk) - 
    4*Fcj*Fcjk*pow(Rj, 2)*Rjk*Rk*eta*(2*zj - zk)*
    pow(CosTheta*Lambda + 1, zeta + 1) + 2.0*Fcj*pow(Rj, 2)*
    Rk*dFcjk*(zj - zk)*pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Fcjk*Rj*Rjk*Rk*dFcj*zj*pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rjk2 + Rk2))/(pow(Rj, 2)*Rjk*Rk*
    (CosTheta*Lambda + 1));
}

void PairNNAngular2::dG4dk(double xj, double yj, double zj, 
               double xk, double yk, double zk, 
               double Rj, double Rk, double Rjk, double CosTheta,
               double eta, double Rc, double zeta, double Lambda,
               double *dGk) {

  double Rj2 = Rj*Rj;
  double Rk2 = Rk*Rk;
  double Rjk2 = Rjk*Rjk;
  double Fcj = Fc(Rj, Rc, false);
  double Fck = Fc(Rk, Rc, false);
  double Fcjk = Fc(Rjk, Rc, true);
  double dFck = dFcdR(Rk, Rc, false);
  double dFcjk = dFcdR(Rjk, Rc, true);

  dGk[0] = pow(2, -zeta)*Fcj*(-2*Fcjk*Fck*Lambda*Rjk*zeta*
    pow(CosTheta*Lambda + 1, zeta)*(CosTheta*Rj*xk - Rk*xj) + 
    4*Fcjk*Fck*Rj*Rjk*pow(Rk, 2)*eta*(xj - 2*xk)*
    pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Fcjk*Rj*Rjk*Rk*dFck*xk*pow(CosTheta*Lambda + 1, zeta + 1) - 
    2.0*Fck*Rj*pow(Rk, 2)*dFcjk*(xj - xk)*
    pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rjk2 + Rk2))/(Rj*Rjk*pow(Rk, 2)*
    (CosTheta*Lambda + 1));

  dGk[1] = pow(2, -zeta)*Fcj*(-2*Fcjk*Fck*Lambda*Rjk*zeta*
    pow(CosTheta*Lambda + 1, zeta)*(CosTheta*Rj*yk - Rk*yj) + 
    4*Fcjk*Fck*Rj*Rjk*pow(Rk, 2)*eta*(yj - 2*yk)*
    pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Fcjk*Rj*Rjk*Rk*dFck*yk*pow(CosTheta*Lambda + 1, zeta + 1) - 
    2.0*Fck*Rj*pow(Rk, 2)*dFcjk*(yj - yk)*
    pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rjk2 + Rk2))/(Rj*Rjk*pow(Rk, 2)*
    (CosTheta*Lambda + 1));

  dGk[2] = pow(2, -zeta)*Fcj*(-2*Fcjk*Fck*Lambda*Rjk*zeta*
    pow(CosTheta*Lambda + 1, zeta)*(CosTheta*Rj*zk - Rk*zj) + 
    4*Fcjk*Fck*Rj*Rjk*pow(Rk, 2)*eta*(zj - 2*zk)*
    pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Fcjk*Rj*Rjk*Rk*dFck*zk*pow(CosTheta*Lambda + 1, zeta + 1) - 
    2.0*Fck*Rj*pow(Rk, 2)*dFcjk*(zj - zk)*
    pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rjk2 + Rk2))/(Rj*Rjk*pow(Rk, 2)*
    (CosTheta*Lambda + 1));
}

void PairNNAngular2::dG5dj(double xj, double yj, double zj, 
            double xk, double yk, double zk, 
            double Rj, double Rk, double CosTheta, 
            double eta, double Rc, double zeta, double Lambda, 
            double *dGj) {

  double Rj2 = Rj*Rj;
  double Rk2 = Rk*Rk;
  double Fcj = Fc(Rj, Rc, false);
  double Fck = Fc(Rk, Rc, false);
  double dFcj = dFcdR(Rj, Rc, false);

  dGj[0] = pow(2, -zeta)*Fck*(-2*Fcj*Lambda*zeta*
    pow(CosTheta*Lambda + 1, zeta)*(CosTheta*Rk*xj - Rj*xk) - 
    4*Fcj*pow(Rj, 2)*Rk*eta*xj*pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Rj*Rk*dFcj*xj*pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rk2))/(pow(Rj, 2)*Rk*(CosTheta*Lambda + 1));

  dGj[1] = pow(2, -zeta)*Fck*(-2*Fcj*Lambda*zeta*
    pow(CosTheta*Lambda + 1, zeta)*(CosTheta*Rk*yj - Rj*yk) - 
    4*Fcj*pow(Rj, 2)*Rk*eta*yj*pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Rj*Rk*dFcj*yj*pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rk2))/(pow(Rj, 2)*Rk*(CosTheta*Lambda + 1));

  dGj[2] = pow(2, -zeta)*Fck*(-2*Fcj*Lambda*zeta*
    pow(CosTheta*Lambda + 1, zeta)*(CosTheta*Rk*zj - Rj*zk) - 
    4*Fcj*pow(Rj, 2)*Rk*eta*zj*pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Rj*Rk*dFcj*zj*pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rk2))/(pow(Rj, 2)*Rk*(CosTheta*Lambda + 1));
}

void PairNNAngular2::dG5dk(double xj, double yj, double zj, 
            double xk, double yk, double zk, 
            double Rj, double Rk, double CosTheta, 
            double eta, double Rc, double zeta, double Lambda, 
            double *dGk) {

  double Rj2 = Rj*Rj;
  double Rk2 = Rk*Rk;
  double Fcj = Fc(Rj, Rc, false);
  double Fck = Fc(Rk, Rc, false);
  double dFck = dFcdR(Rk, Rc, false);

  dGk[0] = pow(2, -zeta)*Fcj*(-2*Fck*Lambda*zeta*
    pow(CosTheta*Lambda + 1, zeta)*(CosTheta*Rj*xk - Rk*xj) - 
    4*Fck*Rj*pow(Rk, 2)*eta*xk*pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Rj*Rk*dFck*xk*pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rk2))/(Rj*pow(Rk, 2)*(CosTheta*Lambda + 1));

  dGk[1] = pow(2, -zeta)*Fcj*(-2*Fck*Lambda*zeta*pow
    (CosTheta*Lambda + 1, zeta)*(CosTheta*Rj*yk - Rk*yj) - 
    4*Fck*Rj*pow(Rk, 2)*eta*yk*pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Rj*Rk*dFck*yk*pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rk2))/(Rj*pow(Rk, 2)*(CosTheta*Lambda + 1));

  dGk[2] = pow(2, -zeta)*Fcj*(-2*Fck*Lambda*zeta*pow
    (CosTheta*Lambda + 1, zeta)*(CosTheta*Rj*zk - Rk*zj) - 
    4*Fck*Rj*pow(Rk, 2)*eta*zk*pow(CosTheta*Lambda + 1, zeta + 1) + 
    2.0*Rj*Rk*dFck*zk*pow(CosTheta*Lambda + 1, zeta + 1))*
    exp(-eta*(Rj2 + Rk2))/(Rj*pow(Rk, 2)*(CosTheta*Lambda + 1));
}


void PairNNAngular2::compute(int eflag, int vflag)
{
  //feenableexcept(FE_INVALID | FE_OVERFLOW);

  double evdwl = 0.0;
  eng_vdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;    // atoms belonging to current processor
  int newton_pair = force->newton_pair; // decides how energy and virial are tallied

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  double fxtmp,fytmp,fztmp;

  int ghosts = 0;
  // loop over full neighbor list of my atoms
  for (int ii = 0; ii < inum; ii++) {
    
    int i = ilist[ii];
    tagint itag = tag[i];

    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];

    // two-body interactions, skip half of them
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];
    int numshort = 0;

    // collect all neighbours in arma matrix, jnum max
    arma::mat Rij(1, jnum);        // all pairs (i,j)
    arma::mat drij(3, jnum);       // (dxij, dyij, dzyij)   
    std::vector<int> tagsj(jnum);  // indicies of j-atoms

    // store all triplets etc in vectors
    // jnum pairs, jnum-1 triplets max
    // for every (i,j) there is a vector of the below quantities
    std::vector<arma::mat> Riks(jnum-1);
    std::vector<arma::mat> driks(jnum-1);
    std::vector<arma::mat> cosThetas(jnum-1);
    std::vector<arma::mat> Rjks(jnum-1);
    std::vector<arma::mat> drjks(jnum-1);
    std::vector<std::vector<int>> tagsk(jnum+3);

    // input vector to NN
    arma::mat inputVector(1, m_numberOfSymmFunc, arma::fill::zeros);

    // keep track of how many atoms below r2
    int neighbours = 0; 

    // collect all pairs
    for (int jj = 0; jj < jnum; jj++) {

      int j = jlist[jj];
      j &= NEIGHMASK;
      tagint jtag = tag[j];

      /*double delxj = xtmp - x[j][0];
      double delyj = ytmp - x[j][1];
      double delzj = ztmp - x[j][2];*/
      double delxj = x[j][0] - xtmp;
      double delyj = x[j][1] - ytmp;
      double delzj = x[j][2] - ztmp;

      double rsq1 = delxj*delxj + delyj*delyj + delzj*delzj;

      if (rsq1 >= cutoff*cutoff) continue;

      // store pair coordinates
      double rij = sqrt(rsq1);
      drij(0, neighbours) = delxj;
      drij(1, neighbours) = delyj;
      drij(2, neighbours) = delzj;
      Rij(0, neighbours) = rij;
      tagsj[neighbours] = j;

      // apply 2-body symmetry
      for (int s=0; s < m_numberOfSymmFunc; s++)
        if ( m_parameters[s].size() == 3 ) 
          inputVector(0,s) += G2(rij, m_parameters[s][0],
                                 m_parameters[s][1], m_parameters[s][2]);

      // collect triplets for this (i,j)
      arma::mat Rik(1, jnum);
      arma::mat drik(3, jnum);
      arma::mat CosTheta(1, jnum);
      arma::mat Rjk(1, jnum); 
      arma::mat drjk(3, jnum);

      // three-body
      int neighk = 0;
      for (int kk = jj+1; kk < jnum; kk++) {

        int k = jlist[kk];
        k &= NEIGHMASK;
        tagint ktag = tag[k];

        //if (k == j) continue;

        /*double delxk = xtmp - x[k][0];
        double delyk = ytmp - x[k][1];
        double delzk = ztmp - x[k][2];*/
        double delxk = x[k][0] - xtmp;
        double delyk = x[k][1] - ytmp;
        double delzk = x[k][2] - ztmp;

        double rsq2 = delxk*delxk + delyk*delyk + delzk*delzk;  

        if (rsq2 >= cutoff*cutoff) continue;
        
        // calculate quantites needed in G4
        double rik = sqrt(rsq2);
        double cosTheta = ( delxj*delxk + delyj*delyk + 
                            delzj*delzk ) / (rij*rik);

        double delxjk = x[j][0] - x[k][0];
        double delyjk = x[j][1] - x[k][1];
        double delzjk = x[j][2] - x[k][2];
        /*double delxjk = x[k][0] - x[j][0];
        double delyjk = x[k][1] - x[j][1];
        double delzjk = x[k][2] - x[j][2];*/
      
        double rjk = sqrt(delxjk*delxjk + delyjk*delyjk + delzjk*delzjk);

        // collect triplets
        drik(0, neighk) = delxk;
        drik(1, neighk) = delyk;
        drik(2, neighk) = delzk;
        Rik(0,neighk) = rik;
        CosTheta(0, neighk) = cosTheta;
        Rjk(0, neighk) = rjk;
        drjk(0, neighk) = delxjk;
        drjk(1, neighk) = delyjk;
        drjk(2, neighk) = delzjk;
        tagsk[neighbours].push_back(k);

        // increment
        neighk++;

        // apply 3-body symmetry
        for (int s=0; s < m_numberOfSymmFunc; s++) 
          if ( m_parameters[s].size() == 4 ) 
            /*inputVector(0,s) += G4(rij, rik, rjk, cosTheta,
                                   m_parameters[s][0], m_parameters[s][1], 
                                   m_parameters[s][2], m_parameters[s][3]);*/
            inputVector(0,s) += G5(rij, rik, cosTheta,
                                   m_parameters[s][0], m_parameters[s][1], 
                                   m_parameters[s][2], m_parameters[s][3]);
      }

      // skip if no triplets left
      if (neighk == 0) {
        neighbours++;
        continue;
      }

      // get rid of empty elements
      Rik = Rik.head_cols(neighk);
      drik = drik.head_cols(neighk);
      CosTheta = CosTheta.head_cols(neighk);
      Rjk = Rjk.head_cols(neighk);
      drjk = drjk.head_cols(neighk);

      // store all k's for current (i,j) to compute forces later
      Riks[neighbours]      = Rik;
      driks[neighbours]     = drik;
      cosThetas[neighbours] = CosTheta;
      Rjks[neighbours]      = Rjk;
      drjks[neighbours]     = drjk;
      neighbours++;
    }

    // get rid of empty elements
    Rij = Rij.head_cols(neighbours);
    drij = drij.head_cols(neighbours);

    // check
    for (auto inputValue : inputVector) {
      if (inputValue > 1.38635211025)
        cout << "Large input value: " << inputValue << endl;
      else if (inputValue < 0.0)
        cout << "Negative input value: " << inputValue << endl;
    }

    // apply NN to get energy
    evdwl = network(inputVector);

    eatom[i] += evdwl/2;///(neighbours);

    // set energy manually (not use ev_tally for energy)
    eng_vdwl += evdwl/2;///(neighbours);

    // backpropagate to obtain gradient of NN
    arma::mat dEdG = backPropagation();

    double fx2 = 0;
    double fy2 = 0;
    double fz2 = 0;

    double fx3j = 0;
    double fy3j = 0;
    double fz3j = 0;
    double fx3k = 0;
    double fy3k = 0;
    double fz3k = 0;
    
    // calculate forces by differentiating the symmetry functions
    // UNTRUE(?): dEdR(j) will be the force contribution from atom j on atom i
    for (int s=0; s < m_numberOfSymmFunc; s++) {
      
      // G2: one atomic pair environment per symmetry function
      if ( m_parameters[s].size() == 3 ) {


        arma::mat dG2(1,neighbours); // derivative of G2

        // calculate cerivative of G2 for all pairs
        // pass dG2 by reference instead of coyping matrices
        // and returning from function --> speed-up
        dG2dR(Rij, m_parameters[s][0],
              m_parameters[s][1], m_parameters[s][2], dG2);

        // chain rule. all pair foces
        arma::mat fpairs = -dEdG(0,s) * dG2 / Rij;

        // loop through all pairs for N3L
        for (int l=0; l < neighbours; l++) {

          double fpair = fpairs(0,l);
          fx2 += fpair*drij(0,l);
          fy2 += fpair*drij(1,l);
          fz2 += fpair*drij(2,l);

          // according to Behler
          f[tagsj[l]][0] += fpair*drij(0,l);
          f[tagsj[l]][1] += fpair*drij(1,l);
          f[tagsj[l]][2] += fpair*drij(2,l);

          if (evflag) ev_tally_full(i, 0, 0, fpair,
                                    drij(0,l), drij(1,l), drij(2,l));
          //if (evflag) ev_tally(i, tagsj[l], nlocal, newton_pair,
          //                     0, 0, fpair,
          //                     drij(0,l), drij(1,l), drij(2,l));
        }
      }

      // G4/G5: neighbours-1 triplet environments per symmetry function
      else {

        for (int l=0; l < neighbours-1; l++) {

          int numberOfTriplets = arma::size(Riks[l])(1);

          arma::mat dEdRj3(3, numberOfTriplets);
          arma::mat dEdRk3(3, numberOfTriplets); // triplet force for all atoms k

          // calculate forces for all triplets (i,j,k) for this (i,j)
          // fj3 and dEdR3 is passed by reference
          // all triplet forces are summed and stored for j in fj3
          // dEdR3 will contain triplet forces for all k, need 
          // each one seperately for N3L
          // Riks[l], Rjks[l], cosThetas[l] : (1,numberOfTriplets)
          // dEdR3, driks[l]: (3, numberOfTriplets)
          // drij.col(l): (1,3)
          /*dG4dR(Rij(0,l), Riks[l], Rjks[l], cosThetas[l],
                m_parameters[s][0], m_parameters[s][1], 
                m_parameters[s][2], m_parameters[s][3], 
                dEdRj3, dEdRk3, drij.col(l), driks[l], drjks[l]);*/

          // N3L: add 3-body forces for i and k
          double fj3[3];
          double fk3[3];
          double dGj[3];
          double dGk[3];
          for (int m=0; m < numberOfTriplets; m++) {

            /*dG4dj(drij(0,l), drij(1,l), drij(2,l), 
              driks[l](0,m), driks[l](1,m), driks[l](2,m), 
              Rij(0,l), Riks[l](0,m), Rjks[l](0,m), 
              cosThetas[l](0,m), m_parameters[s][0], 
              m_parameters[s][1], m_parameters[s][2], 
              m_parameters[s][3], dGj);

            dG4dk(drij(0,l), drij(1,l), drij(2,l), 
              driks[l](0,m), driks[l](1,m), driks[l](2,m), 
              Rij(0,l), Riks[l](0,m), Rjks[l](0,m), 
              cosThetas[l](0,m), m_parameters[s][0], 
              m_parameters[s][1], m_parameters[s][2], 
              m_parameters[s][3], dGk);*/

            dG5dj(drij(0,l), drij(1,l), drij(2,l), 
              driks[l](0,m), driks[l](1,m), driks[l](2,m), 
              Rij(0,l), Riks[l](0,m), cosThetas[l](0,m), 
              m_parameters[s][0], m_parameters[s][1], 
              m_parameters[s][2], m_parameters[s][3], dGj);

            dG5dk(drij(0,l), drij(1,l), drij(2,l), 
              driks[l](0,m), driks[l](1,m), driks[l](2,m), 
              Rij(0,l), Riks[l](0,m), cosThetas[l](0,m), 
              m_parameters[s][0], m_parameters[s][1], 
              m_parameters[s][2], m_parameters[s][3], dGk);

            // triplet force j
            /*fj3[0] = dEdG(0,s) * dEdRj3(0,m);
            fj3[1] = dEdG(0,s) * dEdRj3(1,m);
            fj3[2] = dEdG(0,s) * dEdRj3(2,m);

            // triplet force k
            fk3[0] = dEdG(0,s) * dEdRk3(0,m);
            fk3[1] = dEdG(0,s) * dEdRk3(1,m);
            fk3[2] = dEdG(0,s) * dEdRk3(2,m);*/

            fj3[0] = -dEdG(0,s) * dGj[0];
            fj3[1] = -dEdG(0,s) * dGj[1];
            fj3[2] = -dEdG(0,s) * dGj[2];

            // triplet force k
            fk3[0] = -dEdG(0,s) * dGk[0];
            fk3[1] = -dEdG(0,s) * dGk[1];
            fk3[2] = -dEdG(0,s) * dGk[2];

            // add both j and k to atom i
            fx3j += fj3[0];
            fy3j += fj3[1];
            fz3j += fj3[2];
            fx3k += fk3[0];
            fy3k += fk3[1];
            fz3k += fk3[2];

            // add to atom j. Not N3L, but becuase
            // every pair (i,j) is counted twice for triplets
            f[tagsj[l]][0] += fj3[0];// + fk3[0];
            f[tagsj[l]][1] += fj3[1];// + fk3[1];
            f[tagsj[l]][2] += fj3[2];// + fk3[2];

            // add to atom k 
            f[tagsk[l][m]][0] += fk3[0];// + fj3[0];
            f[tagsk[l][m]][1] += fk3[1];// + fj3[1];
            f[tagsk[l][m]][2] += fk3[2];// + fj3[2];

            if (evflag) ev_tally3_nn(i, tagsj[l], tagsk[l][m],
                                     fj3, fk3, 
                                     drij(0,l), drij(1,l), drij(2,l),
                                     driks[l](0,m), driks[l](1,m), driks[l](2,m));
          }
        }
      }
    }

    // update forces
    /*f[i][0] += fx3j + fx3k;
    f[i][1] += fy3j + fy3k;
    f[i][2] += fz3j + fz3k;*/

    /*f[i][0] -= fx2;
    f[i][1] -= fy2;
    f[i][2] -= fz2;*/
  }

  cout << f[0][0] << " " << f[0][1] << " " << f[0][2] << endl;
  cout << f[1][0] << " " << f[1][1] << " " << f[1][2] << endl;
  cout << f[2][0] << " " << f[2][1] << " " << f[2][2] << endl;
  cout << endl;
  if (myStep == 5) exit(1);

  if (vflag_fdotr) virial_fdotr_compute();

  // write out all forces
  /*std::ofstream forces;
    if (!myStep) {
    forces.open("../TestNN/Tests/Forces/Box/forcesNN.txt");
    forces << "Every 100 steps for 1000 steps" << endl;
  }
  else if (!(myStep % 100))
    forces.open("Tests/Forces/Box/forcesNN.txt", std::ios::app);
  for (int ii; ii < inum; ii++) {
    int i = ilist[ii];
    forces << i << " " << f[i][0] << " " << f[i][1] << " " << 
    f[i][2] << endl;
  }
  forces.close();*/

  myStep++;
}

/* ---------------------------------------------------------------------- */

void PairNNAngular2::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq"); 
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairNNAngular2::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNNAngular2::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg != 4)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read potential file and initialize potential parameters
  read_file(arg[2]);
  cutoff = force->numeric(FLERR,arg[3]);

  // let lammps know that we have set all parameters
  int n = atom->ntypes;
  int count = 0;
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
        setflag[i][j] = 1;
        count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairNNAngular2::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style NN requires atom IDs");
  //if (force->newton_pair == 0)
  //  error->all(FLERR,"Pair style NN requires newton pair on");

  // need a full neighbor list
  int irequest = neighbor->request(this);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairNNAngular2::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutoff;
}

/* ---------------------------------------------------------------------- */

void PairNNAngular2::read_file(char *file)
{
  // convert to string 
  std::string trainingDir(file);
  std::string graphFile = trainingDir + "/graph.dat";
  std::cout << "Graph file: " << graphFile << std::endl;
  
  // open graph file
  std::ifstream inputGraph;
  inputGraph.open(graphFile.c_str(), std::ios::in);

  // check if file successfully opened
  if ( !inputGraph.is_open() ) std::cout << "File is not opened" << std::endl;

  // process first line
  std::string activation;
  inputGraph >> m_nLayers >> m_nNodes >> activation >> 
                m_numberOfInputs >> m_numberOfOutputs;
  std::cout << "Layers: "     << m_nLayers         << std::endl;
  std::cout << "Nodes: "      << m_nNodes          << std::endl;
  std::cout << "Activation: " << activation        << std::endl;
  std::cout << "Neighbours: " << m_numberOfInputs  << std::endl;
  std::cout << "Outputs: "    << m_numberOfOutputs << std::endl;

  // set sizes
  m_preActivations.resize(m_nLayers+2);
  m_activations.resize(m_nLayers+2);
  m_derivatives.resize(m_nLayers+2);

  // skip a blank line
  std::string dummyLine;
  std::getline(inputGraph, dummyLine);

  // process file
  // store all weights in a temporary vector
  // that will be reshaped later
  std::vector<arma::mat> weightsTemp = std::vector<arma::mat>();
  for ( std::string line; std::getline(inputGraph, line); ) {

    if ( line.empty() )
        break;

    // store all weights in a vector
    double buffer;                  // have a buffer string
    std::stringstream ss(line);     // insert the string into a stream

    // while there are new weights on current line, add them to vector
    arma::mat matrix(1,m_nNodes);
    int i = 0;
    while ( ss >> buffer ) {
        matrix(0,i) = buffer;
        i++;
    }
    weightsTemp.push_back(matrix);
  }

  // can put all biases in vector directly
  // no need for temporary vector
  for ( std::string line; std::getline(inputGraph, line); ) {

    // store all weights in vector
    double buffer;                  // have a buffer string
    std::stringstream ss(line);     // insert the string into a stream

    // while there are new weights on current line, add them to vector
    arma::mat matrix(1,m_nNodes);
    int i = 0;
    while ( ss >> buffer ) {
        matrix(0,i) = buffer;
        i++;
    }
    m_biases.push_back(matrix);
  }

  // close file
  inputGraph.close();

  // write out all weights and biases
  /*for (const auto i : weightsTemp)
    std::cout << i << std::endl;
  std::cout << std::endl;
  for (const auto i : m_biases)
    std::cout << i << std::endl;*/

  // resize weights and biases matrices to correct shapes
  m_weights.resize(m_nLayers+1);

  // first hidden layer
  int currentRow = 0;
  m_weights[0]  = weightsTemp[currentRow];
  for (int i=0; i < m_numberOfInputs-1; i++) {
    currentRow++;
    m_weights[0] = arma::join_cols(m_weights[0], weightsTemp[currentRow]);
  }

  // following hidden layers
  for (int i=0; i < m_nLayers-1; i++) {
    currentRow++;
    m_weights[i+1] = weightsTemp[currentRow];
    for (int j=1; j < m_nNodes; j++) {
        currentRow++;
        m_weights[i+1] = arma::join_cols(m_weights[i+1], weightsTemp[currentRow]);
    }
  }

  // output layer
  currentRow++;
  arma::mat outputLayer = weightsTemp[currentRow];
  for (int i=0; i < m_numberOfOutputs-1; i++) {
    currentRow++;
    outputLayer = arma::join_cols(outputLayer, weightsTemp[currentRow]);
  }
  m_weights[m_nLayers] = arma::reshape(outputLayer, m_nNodes, m_numberOfOutputs);

  // reshape bias of output node
  m_biases[m_nLayers].shed_cols(1,m_nNodes-1);

  // obtained transposed matrices
  m_weightsTransposed.resize(m_nLayers+1);
  for (int i=0; i < m_weights.size(); i++)
    m_weightsTransposed[i] = m_weights[i].t();

  // write out entire system for comparison
  /*for (const auto i : m_weights)
    std::cout << i << std::endl;

  for (const auto i : m_biases)
    std::cout << i << std::endl;*/

  // read parameters file
  std::string parametersName = trainingDir + "/parameters.dat";

  std::cout << "Parameters file: " << parametersName << std::endl;

  std::ifstream inputParameters;
  inputParameters.open(parametersName.c_str(), std::ios::in);

  // check if file successfully opened
  if ( !inputParameters.is_open() ) std::cout << "File is not opened" << std::endl;

  inputParameters >> m_numberOfSymmFunc;

  std::cout << "Number of symmetry functions: " << m_numberOfSymmFunc << std::endl;

  // skip blank line
  std::getline(inputParameters, dummyLine);

  int i = 0;
  for ( std::string line; std::getline(inputParameters, line); ) {

    if ( line.empty() )
      break;

    double buffer;                  // have a buffer string
    std::stringstream ss(line);     // insert the string into a stream

    // while there are new parameters on current line, add them to matrix
    m_parameters.resize(m_numberOfSymmFunc);  
    while ( ss >> buffer ) {
        m_parameters[i].push_back(buffer);
    }
    i++;
  }
  inputParameters.close();
  std::cout << "File read......" << std::endl;
}
