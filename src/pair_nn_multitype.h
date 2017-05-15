/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(nn/multitype,PairNNMultiType)

#else

#ifndef LMP_PAIR_NN_MULTITYPE_H
#define LMP_PAIR_NN_MULTITYPE_H

#include "pair.h"

#include <armadillo>
#include <vector>

namespace LAMMPS_NS {

class PairNNMultiType : public Pair {

 public:

  PairNNMultiType(class LAMMPS *);
  virtual ~PairNNMultiType();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void init_style();

  double network(arma::mat inputVector, int type);
  arma::mat backPropagation(int type);
  
  arma::mat sigmoid(arma::mat matrix);
  arma::mat sigmoidDerivative(arma::mat matrix);
  
  arma::mat Fc(arma::mat R, double Rc, bool cut);
  double Fc(double R, double Rc);
  
  arma::mat dFcdR(arma::mat R, double Rc, bool cut);
  double dFcdR(double R, double Rc);
  
  double G1(arma::mat Rij, double Rc);
  double G2(double rij, double eta, double Rc, double Rs);
  double G4(double rij, double rik, double rjk, double cosTheta, 
            double eta, double Rc, double zeta, double lambda);
  double G5(double rij, double rik, double cosTheta, 
                            double eta, double Rc, double zeta, double lambda);

  arma::mat dG1dR(arma::mat Rij, double Rc);

  void dG2dR(arma::mat Rij, double eta, double Rc, double Rs, arma::mat& dG2);

  void dG4dR(double xij, double yij, double zij,
             double xik, double yik, double zik, 
             double xjk, double yjk, double zjk,
             double Rij, double Rik, double Rjk, double cosTheta, 
             double eta, double Rc, double zeta, double lambda,
             double *dGij, double *dGik);

  void dG5dR(double xij, double yij, double zij,
             double xik, double yik, double zik, 
             double Rij, double Rik, double cosTheta, 
             double eta, double Rc, double zeta, double lambda,
             double *dGij, double *dGik);

  void dG4dj(double xj, double yj, double zj, 
               double xk, double yk, double zk, 
               double Rj, double Rk, double Rjk, double CosTheta,
               double eta, double Rc, double zeta, double Lambda,
               double *dGj);

  void dG4dk(double xj, double yj, double zj, 
               double xk, double yk, double zk, 
               double Rj, double Rk, double Rjk, double CosTheta,
               double eta, double Rc, double zeta, double Lambda, 
               double *dGk);

  void dG5dj(double xj, double yj, double zj, 
              double xk, double yk, double zk, 
              double Rj, double Rk, double CosTheta, 
              double eta, double Rc, double zeta, double lambda, 
              double *dGj);

  void dG5dk(double xj, double yj, double zj, 
              double xk, double yk, double zk, 
              double Rj, double Rk, double CosTheta, 
              double eta, double Rc, double zeta, double lambda, 
              double *dGk);
 
 protected:
  double cutoff;
  int nelements;                // # of unique elements
  char **elements;              // names of unique elements
  int ***elem2param;            // mapping from element triplets to parameters
  int *map;                     // mapping from atom types to elements

  std::vector<int> m_nLayers;
  std::vector<int> m_nNodes;
  std::vector<int> m_numberOfInputs;
  std::vector<int> m_numberOfOutputs;

  std::vector<std::vector<arma::mat>> m_weights;
  std::vector<std::vector<arma::mat>> m_weightsTransposed;
  std::vector<std::vector<arma::mat>> m_biases;             
  std::vector<std::vector<arma::mat>> m_preActivations;     
  std::vector<std::vector<arma::mat>> m_activations;        
  std::vector<std::vector<arma::mat>> m_derivatives;        
  
  std::vector<std::vector<std::vector<double>>> m_parameters;
  std::vector<int> m_numberOfSymmFunc;
  std::vector<int> m_numberOfParameters;

  const double m_pi = arma::datum::pi;
  int myStep = 0;

  std::ofstream out;
  std::ofstream out2;

  void allocate();
  void read_file(char *, int type);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style Vashishta requires atom IDs

This is a requirement to use the Vashishta potential.

E: Pair style Vashishta requires newton pair on

See the newton command.  This is a restriction to use the Vashishta
potential.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Cannot open Vashishta potential file %s

The specified Vashishta potential file cannot be opened.  Check that the path
and name are correct.

E: Incorrect format in Vashishta potential file

Incorrect number of words per line in the potential file.

E: Illegal Vashishta parameter

One or more of the coefficients defined in the potential file is
invalid.

E: Potential file has duplicate entry

The potential file has more than one entry for the same element.

E: Potential file is missing an entry

The potential file does not have a needed entry.

*/
