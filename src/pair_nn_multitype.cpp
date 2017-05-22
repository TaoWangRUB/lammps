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
#include "pair_nn_multitype.h"
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
#include <map>
#include <tuple>

#include <fstream>
#include <iomanip>

using namespace LAMMPS_NS;
using std::cout;
using std::endl;

//#define MAXLINE 1024
//#define DELTA 4

/* ---------------------------------------------------------------------- */

PairNNMultiType::PairNNMultiType(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0; // We don't provide the force between two atoms only since it is Angular2
  restartinfo = 0;   // We don't write anything to restart file
  one_coeff = 1;     // only one coeff * * call
  manybody_flag = 1; // Not only a pair style since energies are computed from more than one neighbor
  cutoff = 10.0;      // Will be read from command line

  nelements = 0;
  elements = NULL;
  map = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairNNMultiType::~PairNNMultiType()
{
  if (copymode) return;
  // If you allocate stuff you should delete and deallocate here. 
  // Allocation of normal vectors/matrices (not armadillo), should be created with
  // memory->create(...)

  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] map;
  }
}

/* ---------------------------------------------------------------------- */

double PairNNMultiType::network(arma::mat inputVector, int type) {
    // inputVector has size 1 x inputs

    // linear activation for inputlayer
    m_preActivations[type][0] = inputVector;
    m_activations[type][0] = m_preActivations[type][0];

    // hidden layers
    for (int i=0; i < m_nLayers[type]; i++) {
        // weights and biases starts at first hidden layer:
        // weights[0] are the weights connecting inputGraph layer to first hidden layer
        m_preActivations[type][i+1] = m_activations[type][i]*m_weights[type][i] + 
                                      m_biases[type][i];
        m_activations[type][i+1] = sigmoid(m_preActivations[type][i+1]);
    }

    // linear activation for output layer
    m_preActivations[type][m_nLayers[type]+1] = m_activations[type][m_nLayers[type]]*m_weights[type][m_nLayers[type]] + 
                                          m_biases[type][m_nLayers[type]];
    m_activations[type][m_nLayers[type]+1] = m_preActivations[type][m_nLayers[type]+1];

    // return activation of output neuron
    return m_activations[type][m_nLayers[type]+1](0,0);
}

arma::mat PairNNMultiType::backPropagation(int type) {
  // find derivate of output w.r.t. intput, i.e. dE/dr_ij
  // need to find the "error" terms for all the nodes in all the layers

  // the derivative of the output neuron's activation function w.r.t.
  // its inputGraph is propagated backwards.
  // the output activation function is f(x) = x, so this is 1
  arma::mat output(1,1); output.fill(1);
  m_derivatives[type][m_nLayers[type]+1] = output;

  // we can thus compute the error vectors for the other layers
  for (int i=m_nLayers[type]; i > 0; i--) {
      m_derivatives[type][i] = ( m_derivatives[type][i+1]*m_weightsTransposed[type][i] ) %
                         sigmoidDerivative(m_preActivations[type][i]);
  }

  // linear activation function for inputGraph neurons
  m_derivatives[type][0] = m_derivatives[type][1]*m_weightsTransposed[type][0];

  return m_derivatives[type][0];
}

arma::mat PairNNMultiType::sigmoid(arma::mat matrix) {

  return 1.0/(1 + arma::exp(-matrix));
}

arma::mat PairNNMultiType::sigmoidDerivative(arma::mat matrix) {

  arma::mat sigmoidMatrix = sigmoid(matrix);
  return sigmoidMatrix % (1 - sigmoidMatrix);
}

arma::mat PairNNMultiType::Fc(arma::mat R, double Rc, bool cut) {

  arma::mat value = 0.5*(arma::cos(m_pi*R/Rc) + 1);

  if (cut)
    for (int i=0; i < arma::size(R)(1); i++)
      if (R(0,i) > Rc) 
        value(0,i) = 0;
  
  return value;
}

double PairNNMultiType::Fc(double R, double Rc) {

  if (R < Rc)
    return 0.5*(cos(m_pi*R/Rc) + 1);
  else
    return 0;
}

arma::mat PairNNMultiType::dFcdR(arma::mat R, double Rc, bool cut) {

  double Rcinv = 1.0/Rc;
  arma::mat value = -(0.5*m_pi*Rcinv) * arma::sin(m_pi*R*Rcinv);

  if (cut)
    for (int i=0; i < arma::size(R)(1); i++)
      if (R(0,i) > Rc) 
        value(0,i) = 0;
  
  return value; 
}

double PairNNMultiType::dFcdR(double R, double Rc) {

  if (R < Rc) {
    double Rcinv = 1.0/Rc;
    return -(0.5*m_pi*Rcinv) * sin(m_pi*R*Rcinv);
  }
  else return 0;
}

double PairNNMultiType::G1(arma::mat Rij, double Rc) {

  return arma::accu( Fc(Rij, Rc, false) );
}

arma::mat PairNNMultiType::dG1dR(arma::mat Rij, double Rc) {

  return dFcdR(Rij, Rc, false);
}

double PairNNMultiType::G2(double rij, double eta, double Rc, double Rs) {

  return exp(-eta*(rij - Rs)*(rij - Rs)) * Fc(rij, Rc);
}

void PairNNMultiType::dG2dR(arma::mat Rij, double eta, double Rc, double Rs,
                          arma::mat &dG2) {

  dG2 = arma::exp(-eta*(Rij - Rs)%(Rij - Rs)) % 
          ( 2*eta*(Rs - Rij) % Fc(Rij, Rc, false) + dFcdR(Rij, Rc, false) );
}

double PairNNMultiType::G4(double rij, double rik, double rjk, double cosTheta, 
                          double eta, double Rc, double zeta, double lambda) {

  return pow(2, 1-zeta) * 
         pow(1 + lambda*cosTheta, zeta) *  
         exp( -eta*(rij*rij + rik*rik + rjk*rjk) ) * 
         Fc(rij, Rc) * Fc(rik, Rc) * Fc(rjk, Rc);
}

double PairNNMultiType::G5(double rij, double rik, double cosTheta, 
                          double eta, double Rc, double zeta, double lambda) {

  return pow(2, 1-zeta) * 
         pow(1 + lambda*cosTheta, zeta) *  
         exp( -eta*(rij*rij + rik*rik) ) * 
         Fc(rij, Rc) * Fc(rik, Rc);
}


void PairNNMultiType::dG4dR(double xij, double yij, double zij,
                           double xik, double yik, double zik, 
                           double xjk, double yjk, double zjk,
                           double Rij, double Rik, double Rjk, double cosTheta, 
                           double eta, double Rc, double zeta, double lambda,
                           double *dGij, double *dGik) {

  double powCosThetaM1 = pow(2, 1-zeta)*pow(1 + lambda*cosTheta, zeta-1);
  double F1 = powCosThetaM1 * (1 + lambda*cosTheta);

  double F2 = exp(-eta*(Rij*Rij + Rik*Rik + Rjk*Rjk));

  double FcRij = Fc(Rij, Rc);
  double FcRik = Fc(Rik, Rc);
  double FcRjk = Fc(Rjk, Rc);
  double F3 = FcRij * FcRik * FcRjk;

  double K = lambda*zeta*powCosThetaM1;
  double L = 2*eta*F2;
  double Mij = dFcdR(Rij, Rc);
  double Mik = dFcdR(Rik, Rc);
  double Mjk = dFcdR(Rjk, Rc);

  double term1 = K * F2 * F3;
  double term2 = F1 * L * F3;

  double F1F2 = F1 * F2;

  double term3ij = F1F2 * Mij * FcRik * FcRjk;
  double term3ik = F1F2 * Mik * FcRij * FcRjk;

  double termjkij = (F1F2 * Mjk * FcRij * FcRik) + 2*F1F2*F3;
  double termjkik = -(F1F2 * Mjk * FcRij * FcRik) + 2*F1F2*F3;

  double RijInv = 1.0 / Rij;
  double cosRijInv2 = cosTheta * RijInv*RijInv;

  double RikInv = 1.0 / Rik;
  double cosRikInv2 = cosTheta * RikInv*RikInv;

  double RjkInv = 1.0 / Rjk;

  double RijRikInv = RijInv * RikInv;

  double termij = (cosRijInv2 * term1) + term2 - RijInv*term3ij;
  double termik = (cosRikInv2 * term1) + term2 - RikInv*term3ik;
  double crossTerm = -term1 * RijRikInv;
  termjkij *= RjkInv;                 
  termjkik *= RjkInv;

  dGij[0] = xij*termij + xik*crossTerm + xjk*termjkij;
  dGij[1] = yij*termij + yik*crossTerm + yjk*termjkij;
  dGij[2] = zij*termij + zik*crossTerm + zjk*termjkij;

  dGik[0] = xik*termik + xij*crossTerm + xjk*termjkik;
  dGik[1] = yik*termij + yij*crossTerm + yjk*termjkik;
  dGik[2] = zik*termij + zij*crossTerm + zjk*termjkik;
}

void PairNNMultiType::dG5dR(double xij, double yij, double zij,
                           double xik, double yik, double zik, 
                           double Rij, double Rik, double cosTheta, 
                           double eta, double Rc, double zeta, double lambda,
                           double *dGij, double *dGik) {

  double powCosThetaM1 = pow(2, 1-zeta)*pow(1 + lambda*cosTheta, zeta-1);
  double F1 = powCosThetaM1 * (1 + lambda*cosTheta);

  double F2 = exp(-eta*(Rij*Rij + Rik*Rik));

  double FcRij = Fc(Rij, Rc);
  double FcRik = Fc(Rik, Rc);
  double F3 = FcRij * FcRik;

  double K = lambda*zeta*powCosThetaM1;
  double L = 2*eta*F2;
  double Mij = dFcdR(Rij, Rc);
  double Mik = dFcdR(Rik, Rc);

  double term1 = K * F2 * F3;
  double term2 = F1 * L * F3;

  double F1F2 = F1 * F2;

  double term3ij = F1F2 * Mij * FcRik;
  double term3ik = F1F2 * Mik * FcRij;

  double RijInv = 1.0 / Rij;
  double cosRijInv2 = cosTheta * RijInv*RijInv;

  double RikInv = 1.0 / Rik;
  double cosRikInv2 = cosTheta * RikInv*RikInv;

  double RijRikInv = RijInv * RikInv;

  double termij = (cosRijInv2 * term1) + term2 - RijInv*term3ij;
  double termik = (cosRikInv2 * term1) + term2 - RikInv*term3ik;
  double crossTerm = -term1 * RijRikInv;

  dGij[0] = -xij*termij - xik*crossTerm;
  dGij[1] = -yij*termij - yik*crossTerm;
  dGij[2] = -zij*termij - zik*crossTerm;

  dGik[0] = -xik*termik - xij*crossTerm;
  dGik[1] = -yik*termik - yij*crossTerm;
  dGik[2] = -zik*termik - zij*crossTerm;
}

void PairNNMultiType::dG4dj(double xj, double yj, double zj, 
               double xk, double yk, double zk, 
               double Rj, double Rk, double Rjk, double CosTheta,
               double eta, double Rc, double zeta, double Lambda, 
               double *dGj) {

  double Rj2 = Rj*Rj;
  double Rk2 = Rk*Rk;
  double Rjk2 = Rjk*Rjk;
  double Fcj = Fc(Rj, Rc);
  double Fck = Fc(Rk, Rc);
  double Fcjk = Fc(Rjk, Rc);
  double dFcj = dFcdR(Rj, Rc);
  double dFcjk = dFcdR(Rjk, Rc);

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

void PairNNMultiType::dG4dk(double xj, double yj, double zj, 
               double xk, double yk, double zk, 
               double Rj, double Rk, double Rjk, double CosTheta,
               double eta, double Rc, double zeta, double Lambda,
               double *dGk) {

  double Rj2 = Rj*Rj;
  double Rk2 = Rk*Rk;
  double Rjk2 = Rjk*Rjk;
  double Fcj = Fc(Rj, Rc);
  double Fck = Fc(Rk, Rc);
  double Fcjk = Fc(Rjk, Rc);
  double dFck = dFcdR(Rk, Rc);
  double dFcjk = dFcdR(Rjk, Rc);

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

/*void PairNNMultiType::dG4dj(double xj, double yj, double zj, 
                           double xk, double yk, double zk, 
                           double Rj, double Rk, double Rjk, double CosTheta,
                           double eta, double Rc, double zeta, double Lambda, 
                           double *dGj) {

  double Rj2 = Rj*Rj;
  double Rk2 = Rk*Rk;
  double Rjk2 = Rjk*Rjk;
  double Fcj = Fc(Rj, Rc);
  double Fck = Fc(Rk, Rc);
  double Fcjk = Fc(Rjk, Rc);
  double dFcj = dFcdR(Rj, Rc);
  double dFcjk = dFcdR(Rjk, Rc);
  double expR = exp(-eta*(Rj2 + Rjk2 + Rk2));
  double powZeta = pow(2, -zeta);
  double powCosTheta = pow(CosTheta*Lambda + 1, zeta);
  double powCosThetaPlus1 = powCosTheta*(CosTheta*Lambda + 1);

  dGj[0] = powZeta*Fck*(-2*Fcj*Fcjk*Lambda*Rjk*zeta*
    powCosTheta*(CosTheta*Rk*xj - Rj*xk) - 
    4*Fcj*Fcjk*Rj2*Rjk*Rk*eta*(2*xj - xk)*
    powCosThetaPlus1 + 2.0*Fcj*Rj2*
    Rk*dFcjk*(xj - xk)*powCosThetaPlus1 + 
    2.0*Fcjk*Rj*Rjk*Rk*dFcj*xj*powCosThetaPlus1)*
    expR/(Rj2*Rjk*Rk*
    (CosTheta*Lambda + 1));

  dGj[1] = powZeta*Fck*(-2*Fcj*Fcjk*Lambda*Rjk*zeta*
    powCosTheta*(CosTheta*Rk*yj - Rj*yk) - 
    4*Fcj*Fcjk*Rj2*Rjk*Rk*eta*(2*yj - yk)*
    powCosThetaPlus1 + 2.0*Fcj*Rj2*
    Rk*dFcjk*(yj - yk)*powCosThetaPlus1 + 
    2.0*Fcjk*Rj*Rjk*Rk*dFcj*yj*powCosThetaPlus1)*
    expR/(Rj2*Rjk*Rk*
    (CosTheta*Lambda + 1));

  dGj[2] = powZeta*Fck*(-2*Fcj*Fcjk*Lambda*Rjk*zeta*
    powCosTheta*(CosTheta*Rk*zj - Rj*zk) - 
    4*Fcj*Fcjk*Rj2*Rjk*Rk*eta*(2*zj - zk)*
    powCosThetaPlus1 + 2.0*Fcj*Rj2*
    Rk*dFcjk*(zj - zk)*powCosThetaPlus1 + 
    2.0*Fcjk*Rj*Rjk*Rk*dFcj*zj*powCosThetaPlus1)*
    expR/(Rj2*Rjk*Rk*
    (CosTheta*Lambda + 1));
}

void PairNNMultiType::dG4dk(double xj, double yj, double zj, 
                           double xk, double yk, double zk, 
                           double Rj, double Rk, double Rjk, double CosTheta,
                           double eta, double Rc, double zeta, double Lambda,
                           double *dGk) {

  double Rj2 = Rj*Rj;
  double Rk2 = Rk*Rk;
  double Rjk2 = Rjk*Rjk;
  double Fcj = Fc(Rj, Rc);
  double Fck = Fc(Rk, Rc);
  double Fcjk = Fc(Rjk, Rc);
  double dFck = dFcdR(Rk, Rc);
  double dFcjk = dFcdR(Rjk, Rc);
  double expR = exp(-eta*(Rj2 + Rjk2 + Rk2));
  double powZeta = pow(2, -zeta);
  double powCosTheta = pow(CosTheta*Lambda + 1, zeta);
  double powCosThetaPlus1 = powCosTheta*(CosTheta*Lambda + 1);

  dGk[0] = powZeta*Fcj*(-2*Fcjk*Fck*Lambda*Rjk*zeta*
    powCosTheta*(CosTheta*Rj*xk - Rk*xj) + 
    4*Fcjk*Fck*Rj*Rjk*Rk2*eta*(xj - 2*xk)*
    powCosThetaPlus1 + 
    2.0*Fcjk*Rj*Rjk*Rk*dFck*xk*powCosThetaPlus1 - 
    2.0*Fck*Rj*Rk2*dFcjk*(xj - xk)*
    powCosThetaPlus1)*
    expR/(Rj*Rjk*Rk2*
    (CosTheta*Lambda + 1));

  dGk[1] = powZeta*Fcj*(-2*Fcjk*Fck*Lambda*Rjk*zeta*
    powCosTheta*(CosTheta*Rj*yk - Rk*yj) + 
    4*Fcjk*Fck*Rj*Rjk*Rk2*eta*(yj - 2*yk)*
    powCosThetaPlus1 + 
    2.0*Fcjk*Rj*Rjk*Rk*dFck*yk*powCosThetaPlus1 - 
    2.0*Fck*Rj*Rk2*dFcjk*(yj - yk)*
    powCosThetaPlus1)*
    expR/(Rj*Rjk*Rk2*
    (CosTheta*Lambda + 1));

  dGk[2] = powZeta*Fcj*(-2*Fcjk*Fck*Lambda*Rjk*zeta*
    powCosTheta*(CosTheta*Rj*zk - Rk*zj) + 
    4*Fcjk*Fck*Rj*Rjk*Rk2*eta*(zj - 2*zk)*
    powCosThetaPlus1 + 
    2.0*Fcjk*Rj*Rjk*Rk*dFck*zk*powCosThetaPlus1 - 
    2.0*Fck*Rj*Rk2*dFcjk*(zj - zk)*
    powCosThetaPlus1)*
    expR/(Rj*Rjk*Rk2*
    (CosTheta*Lambda + 1));
}*/


void PairNNMultiType::compute(int eflag, int vflag)
{
  //feenableexcept(FE_INVALID | FE_OVERFLOW);

  double evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;    // atoms belonging to current processor
  int newton_pair = force->newton_pair; // decides how energy and virial are tallied

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  double fxtmp,fytmp,fztmp;

  // loop over full neighbor list of my atoms
  for (int ii = 0; ii < inum; ii++) {
    
    int i = ilist[ii];
    tagint itag = tag[i];
    int itype = map[type[i]];

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
    std::vector<int> tagsj;  // indicies of j-atoms
    std::vector<int> typesj; // types of j-atoms

    // store all triplets etc in vectors
    // jnum pairs, jnum-1 triplets max
    // for every (i,j) there is a vector of the below quantities
    std::vector<arma::mat> Riks(jnum-1);
    std::vector<arma::mat> driks(jnum-1);
    std::vector<arma::mat> cosThetas(jnum-1);
    std::vector<arma::mat> Rjks(jnum-1);
    std::vector<arma::mat> drjks(jnum-1);
    std::vector<std::vector<int>> tagsk;
    std::vector<std::vector<int>> typesk;

    // input vector to NN
    arma::mat inputVector(1, m_numberOfSymmFunc[itype], arma::fill::zeros);

    // keep track of how many atoms below r2
    int neighbours = 0;

    // collect all pairs
    for (int jj = 0; jj < jnum; jj++) {

      int j = jlist[jj];
      j &= NEIGHMASK;
      tagint jtag = tag[j];
      int jtype = map[type[j]];

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
      tagsj.push_back(j);
      typesj.push_back(jtype);
      tagsk.push_back(std::vector<int>());
      typesk.push_back(std::vector<int>());

      // apply 2-body symmetry
      int a, b, n;
      std::tie(a, b) = elem2param2[std::make_pair(itype,jtype)];
      n = b - a;
      arma::ivec rangeList2 = arma::linspace<arma::ivec>(a, b-1, n);
      for (auto s : rangeList2)
        if ( m_parameters[itype][s].size() == 3 ) 
          inputVector(0,s) += G2(rij, m_parameters[itype][s][0],
                                      m_parameters[itype][s][1], 
                                      m_parameters[itype][s][2]);

      if (rsq1 >= 2.6*2.6) {
        neighbours++;
        continue;
      }

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
        int ktype = map[type[k]];

        double delxk = x[k][0] - xtmp;
        double delyk = x[k][1] - ytmp;
        double delzk = x[k][2] - ztmp;

        double rsq2 = delxk*delxk + delyk*delyk + delzk*delzk;  

        if (rsq2 >= 2.6*2.6) continue;
        
        // calculate quantites needed in G4/G5
        double rik = sqrt(rsq2);
        double cosTheta = ( delxj*delxk + delyj*delyk + 
                            delzj*delzk ) / (rij*rik);

        double delxjk = x[k][0] - x[j][0];
        double delyjk = x[k][1] - x[j][1];
        double delzjk = x[k][2] - x[j][2];
      
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
        typesk[neighbours].push_back(ktype);

        // increment
        neighk++;

        // apply 3-body symmetry
        std::tie(a, b) = elem2param3[std::make_tuple(itype,jtype,ktype)];
        n = b - a;
        arma::ivec rangeList3 = arma::linspace<arma::ivec>(a, b-1, n);
        for (auto s : rangeList3)
          if ( m_parameters[itype][s].size() == 4 ) 
            /*inputVector(0,s) += G4(rij, rik, rjk, cosTheta,
                                   m_parameters[s][0], m_parameters[s][1], 
                                   m_parameters[s][2], m_parameters[s][3]);*/
            inputVector(0,s) += G5(rij, rik, cosTheta,
                                   m_parameters[itype][s][0], 
                                   m_parameters[itype][s][1], 
                                   m_parameters[itype][s][2], 
                                   m_parameters[itype][s][3]);
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
      if (inputValue > 14.6962)
        cout << "Large input value: " << inputValue << endl;
      else if (inputValue < 0.0)
        cout << "Negative input value: " << inputValue << endl;
    }

    // apply NN to get energy
    evdwl = network(inputVector, itype);

    eatom[i] += evdwl;

    // set energy manually (not use ev_tally for energy)
    eng_vdwl += evdwl;

    // backpropagate to obtain gradient of NN
    arma::mat dEdG = backPropagation(itype);

    if (myStep >= 110000) {
      cout << std::setprecision(17) << endl;
      cout << "itype: " << i << endl;
      inputVector.raw_print(cout);
      cout << std::setprecision(17) << "energy: " << evdwl << endl;
      dEdG.raw_print(cout);
      cout << Rij << endl;
      if (myStep == 40) exit(1);
    }

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
    for (int s=0; s < m_numberOfSymmFunc[itype]; s++) {
      
      // G2: one atomic pair environment per symmetry function
      if ( m_parameters[itype][s].size() == 3 ) {

        arma::mat dG2(1,neighbours); // derivative of G2

        // calculate cerivative of G2 for all pairs
        // pass dG2 by reference instead of coyping matrices
        // and returning from function --> speed-up
        dG2dR(Rij, m_parameters[itype][s][0],
              m_parameters[itype][s][1], m_parameters[itype][s][2], dG2);

        // chain rule. all pair foces
        arma::mat fpairs = -dEdG(0,s) * dG2 / Rij;

        // loop through all pairs for N3L
        for (int l=0; l < neighbours; l++) {

          double fpair = fpairs(0,l);
  
          fx2 -= fpair*drij(0,l);
          fy2 -= fpair*drij(1,l);
          fz2 -= fpair*drij(2,l);

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

          // calculate forces for all triplets (i,j,k) for this (i,j)
          // fj3 and dEdR3 is passed by reference
          // all triplet forces are summed and stored for j in fj3
          // dEdR3 will contain triplet forces for all k, need 
          // each one seperately for N3L
          // Riks[l], Rjks[l], cosThetas[l] : (1,numberOfTriplets)
          // dEdR3, driks[l]: (3, numberOfTriplets)
          // drij.col(l): (1,3)

          // N3L: add 3-body forces for i and k
          double fj3[3];
          double fk3[3];
          double dGj[3];
          double dGk[3];
          for (int m=0; m < numberOfTriplets; m++) {

            /*dG4dR(drij(0,l), drij(1,l), drij(2,l),
                  driks[l](0,m), driks[l](1,m), driks[l](2,m),
                  drjks[l](0,m), drjks[l](1,m), drjks[l](2,m),
                  Rij(0,l), Riks[l](0,m), Rjks[l](0,m), cosThetas[l](0,m),
                  m_parameters[s][0], m_parameters[s][1], 
                  m_parameters[s][2], m_parameters[s][3], 
                  dGj, dGk);*/


            dG5dR(drij(0,l), drij(1,l), drij(2,l),
                  driks[l](0,m), driks[l](1,m), driks[l](2,m),
                  Rij(0,l), Riks[l](0,m), cosThetas[l](0,m),
                  m_parameters[itype][s][0], m_parameters[itype][s][1], 
                  m_parameters[itype][s][2], m_parameters[itype][s][3], 
                  dGj, dGk);

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

            fj3[0] = -dEdG(0,s) * dGj[0];
            fj3[1] = -dEdG(0,s) * dGj[1];
            fj3[2] = -dEdG(0,s) * dGj[2];

            // triplet force k
            fk3[0] = -dEdG(0,s) * dGk[0];
            fk3[1] = -dEdG(0,s) * dGk[1];
            fk3[2] = -dEdG(0,s) * dGk[2];

            // add both j and k to atom i  
            fx3j -= fj3[0];
            fy3j -= fj3[1];
            fz3j -= fj3[2];
            fx3k -= fk3[0];
            fy3k -= fk3[1];
            fz3k -= fk3[2];

            // add to atom j. Not N3L, but becuase
            // every pair (i,j) is counted twice for triplets
            f[tagsj[l]][0] += fj3[0];
            f[tagsj[l]][1] += fj3[1];
            f[tagsj[l]][2] += fj3[2];

            // add to atom k 
            f[tagsk[l][m]][0] += fk3[0];
            f[tagsk[l][m]][1] += fk3[1];
            f[tagsk[l][m]][2] += fk3[2];

            if (evflag) ev_tally3_nn(i, tagsj[l], tagsk[l][m],
                                     fj3, fk3, 
                                     drij(0,l), drij(1,l), drij(2,l),
                                     driks[l](0,m), driks[l](1,m), driks[l](2,m));
          }
        }
      }
    }

    // update forces
    f[i][0] += fx3j + fx3k;
    f[i][1] += fy3j + fy3k;
    f[i][2] += fz3j + fz3k;

    f[i][0] += fx2;
    f[i][1] += fy2;
    f[i][2] += fz2;
  }

  if (vflag_fdotr) virial_fdotr_compute();

  myStep++;
}

/* ---------------------------------------------------------------------- */

void PairNNMultiType::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq"); 

  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairNNMultiType::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNNMultiType::coeff(int narg, char **arg)
{
  int i, j, n;

  if (!allocated) allocate();

  if (narg != 5 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // make map
  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }
  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;

  nelements = 0;
  for (i = 4; i < narg-1; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-3] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i-3] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }

  n = atom->ntypes;

  // set sizes according to number of types
  // m_weights[i][j] will be all weights connecting
  // layer j and j+1 of NN of type i
  m_nLayers.resize(n);
  m_nNodes.resize(n);
  m_numberOfInputs.resize(n);
  m_numberOfOutputs.resize(n);
  m_weights.resize(n);
  m_biases.resize(n);
  m_weightsTransposed.resize(n);
  m_preActivations.resize(n);
  m_activations.resize(n);
  m_derivatives.resize(n);
  m_parameters.resize(n);
  m_numberOfSymmFunc.resize(n);

  // read graph files
  for (int i=0; i < n; i++)
    read_file(arg[2+i], i);

  cutoff = force->numeric(FLERR,arg[6]);

  // let lammps know that we have set all parameters
  
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

void PairNNMultiType::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style NN requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style NN requires newton pair on");

  // need a full neighbor list
  int irequest = neighbor->request(this);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairNNMultiType::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutoff;
}

/* ---------------------------------------------------------------------- */

void PairNNMultiType::read_file(char *file, int type)
{
  cout << "Type: " << type << endl;

  // convert to string 
  std::string trainingDir(file);
  std::string graphFile = trainingDir + "/graph.dat";
  std::cout << "Graph file: " << graphFile << std::endl;
  
  // open graph file
  std::ifstream inputGraph;
  inputGraph.open(graphFile.c_str(), std::ios::in);
  if ( !inputGraph.is_open() ) std::cout << "File is not opened" << std::endl;

  // process first line
  int tmpLayers, tmpNodes, tmpInputs, tmpOutputs;
  std::string activation;
  inputGraph >> tmpLayers >> tmpNodes >> activation >> 
                tmpInputs >> tmpOutputs;

  m_nLayers[type] = tmpLayers;
  m_nNodes[type] = tmpNodes;
  m_numberOfInputs[type] = tmpInputs;
  m_numberOfOutputs[type] = tmpOutputs;

  std::cout << "Layers: "     << m_nLayers[type]         << std::endl;
  std::cout << "Nodes: "      << m_nNodes[type]          << std::endl;
  std::cout << "Activation: " << activation              << std::endl;
  std::cout << "Neighbours: " << m_numberOfInputs[type]  << std::endl;
  std::cout << "Outputs: "    << m_numberOfOutputs[type] << std::endl;

  // set sizes
  m_preActivations[type].resize(m_nLayers[type]+2);
  m_activations[type].resize(m_nLayers[type]+2);
  m_derivatives[type].resize(m_nLayers[type]+2);

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
    arma::mat matrix(1,m_nNodes[type]);
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
    arma::mat matrix(1,m_nNodes[type]);
    int i = 0;
    while ( ss >> buffer ) {
        matrix(0,i) = buffer;
        i++;
    }
    m_biases[type].push_back(matrix);
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
  m_weights[type].resize(m_nLayers[type]+1);

  // first hidden layer
  int currentRow = 0;
  m_weights[type][0]  = weightsTemp[currentRow];
  for (int i=0; i < m_numberOfInputs[type]-1; i++) {
    currentRow++;
    m_weights[type][0] = arma::join_cols(m_weights[type][0], 
                                         weightsTemp[currentRow]);
  }

  // following hidden layers
  for (int i=0; i < m_nLayers[type]-1; i++) {
    currentRow++;
    m_weights[type][i+1] = weightsTemp[currentRow];
    for (int j=1; j < m_nNodes[type]; j++) {
        currentRow++;
        m_weights[type][i+1] = arma::join_cols(m_weights[type][i+1], 
                                               weightsTemp[currentRow]);
    }
  }

  // output layer
  currentRow++;
  arma::mat outputLayer = weightsTemp[currentRow];
  for (int i=0; i < m_numberOfOutputs[type]-1; i++) {
    currentRow++;
    outputLayer = arma::join_cols(outputLayer, weightsTemp[currentRow]);
  }
  m_weights[type][m_nLayers[type]] = arma::reshape(outputLayer, 
                                             m_nNodes[type], 
                                             m_numberOfOutputs[type]);

  // reshape bias of output node
  m_biases[type][m_nLayers[type]].shed_cols(1,m_nNodes[type]-1);

  // obtained transposed matrices
  m_weightsTransposed[type].resize(m_nLayers[type]+1);
  for (int i=0; i < m_weights[type].size(); i++)
    m_weightsTransposed[type][i] = m_weights[type][i].t();

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

  inputParameters >> m_numberOfSymmFunc[type];
  std::cout << "Number of symmetry functions: " << m_numberOfSymmFunc[type] << std::endl;

  // skip blank line
  std::getline(inputParameters, dummyLine);

  // make mapping from element to parameter indicies

  // G2
  int nTypes = atom->ntypes;
  std::vector<std::pair<int,int>> pairs(nTypes);
  for (int j=0; j < nTypes; j++)
    pairs[j] = std::make_pair(type,j);

  // G4
  std::vector<std::tuple<int,int,int>> triplets;
  for (int j=0; j < nTypes; j++)
    for (int k=0; k < nTypes; k++)
      triplets.push_back(std::make_tuple(type,j,k));

  int sOld = 0;
  int sNew = 0;
  int i = 0;
  for ( std::string line; std::getline(inputParameters, line); ) {

    if ( line.empty() ) {
      if (i < nTypes)
        elem2param2[std::make_pair(std::get<0>(pairs[i]), std::get<1>(pairs[i]))] = 
          std::make_pair(sOld,sNew);
      else {
        int i3 = i - nTypes;
        elem2param3[std::make_tuple(std::get<0>(triplets[i3]), 
                           std::get<1>(triplets[i3]), 
                           std::get<2>(triplets[i3]))] = 
          std::make_pair(sOld,sNew);
      }
      sOld = sNew;
      i++;
      continue;
    }

    double buffer;                  // have a buffer string
    std::stringstream ss(line);     // insert the string into a stream

    // while there are new parameters on current line, add them to matrix
    m_parameters[type].resize(m_numberOfSymmFunc[type]); 
    while ( ss >> buffer ) {
        m_parameters[type][sNew].push_back(buffer);
    }
    sNew++;
  }
  inputParameters.close();
  std::cout << "Parameters file read......" << std::endl;

  if (type == 1) {
    for (auto i: m_parameters)
      for (auto j : i) {
        for (auto k : j)
          cout << k << " ";
        cout << endl;
      }
  } 

  cout << std::get<0>(elem2param2[std::make_pair(type,0)]) << endl;
  cout << std::get<1>(elem2param2[std::make_pair(type,0)]) << endl;
  cout << std::get<0>(elem2param2[std::make_pair(type,1)]) << endl;
  cout << std::get<1>(elem2param2[std::make_pair(type,1)]) << endl;

  cout << std::get<0>(elem2param3[std::make_tuple(type,0,0)]) << endl;
  cout << std::get<1>(elem2param3[std::make_tuple(type,0,0)]) << endl;
  cout << std::get<0>(elem2param3[std::make_tuple(type,0,1)]) << endl;
  cout << std::get<1>(elem2param3[std::make_tuple(type,0,1)]) << endl;
  cout << std::get<0>(elem2param3[std::make_tuple(type,1,0)]) << endl;
  cout << std::get<1>(elem2param3[std::make_tuple(type,1,0)]) << endl;
  cout << std::get<0>(elem2param3[std::make_tuple(type,1,1)]) << endl;
  cout << std::get<1>(elem2param3[std::make_tuple(type,1,1)]) << endl;
}
