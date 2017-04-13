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

PairStyle(nn/angular2,PairNNAngular2)

#else

#ifndef LMP_PAIR_NN_ANGULAR2_H
#define LMP_PAIR_NN_ANGULAR2_H

#include "pair.h"

#include <armadillo>
#include <vector>

namespace LAMMPS_NS {

class PairNNAngular2 : public Pair {
 public:
  PairNNAngular2(class LAMMPS *);
  virtual ~PairNNAngular2();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void init_style();

  double network(arma::mat inputVector);
  arma::mat backPropagation();
  
  arma::mat sigmoid(arma::mat matrix);
  arma::mat sigmoidDerivative(arma::mat matrix);
  
  arma::mat Fc(arma::mat R, double Rc, bool cut);
  double Fc(double R, double Rc, bool cut);
  
  arma::mat dFcdR(arma::mat R, double Rc, bool cut);
  double dFcdR(double R, double Rc, bool cut);
  
  double G1(arma::mat Rij, double Rc);
  double G2(double rij, double eta, double Rc, double Rs);
  double G4(double rij, double rik, double rjk, double cosTheta, 
            double eta, double Rc, double zeta, double lambda);
  double G5(double rij, double rik, double cosTheta, 
                            double eta, double Rc, double zeta, double lambda);

  arma::mat dG1dR(arma::mat Rij, double Rc);
  void dG2dR(arma::mat Rij, double eta, double Rc, double Rs, arma::mat& dG2);
  void dG4dR(double Rij, arma::mat Rik, arma::mat Rjk, 
             arma::mat cosTheta, double eta, double Rc, 
             double zeta, double lambda,
             arma::mat &dEdRj3, arma::mat &dEdRk3,
             arma::mat drij, arma::mat drik, arma::mat drjk);
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

  int m_nLayers;
  int m_nNodes;
  std::vector<arma::mat> m_weights            = std::vector<arma::mat>();
  std::vector<arma::mat> m_weightsTransposed  = std::vector<arma::mat>();
  std::vector<arma::mat> m_biases             = std::vector<arma::mat>();
  std::vector<arma::mat> m_preActivations     = std::vector<arma::mat>();
  std::vector<arma::mat> m_activations        = std::vector<arma::mat>();
  std::vector<arma::mat> m_derivatives        = std::vector<arma::mat>();
  std::vector<std::vector<double>> m_parameters;
  int m_numberOfInputs;
  int m_numberOfOutputs;
  int m_numberOfSymmFunc;
  int m_numberOfParameters;
  const double m_pi = arma::datum::pi;
  int myStep = 0;
  std::ofstream pairForces;
  std::ofstream tripletForces;

  arma::mat configs;

  void allocate();
  void read_file(char *);
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
