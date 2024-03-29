% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/buildPredMat.R
\name{buildPredMat}
\alias{buildPredMat}
\title{Build a prediction matrix from CV model fits}
\usage{
buildPredMat(
  cvfitlist,
  y,
  lambda,
  family,
  foldid,
  predict_fun,
  predict_params,
  predict_row_params = c(),
  type.measure = NULL,
  weights = NULL,
  grouped = NULL
)
}
\arguments{
\item{cvfitlist}{A list of length `nfolds`, with each element being
the model fit for each fold.}

\item{y}{Response. It is only used to determine what dimensions the
prediction array needs to have.}

\item{lambda}{Lambda values for which we want predictions.}

\item{family}{Model family; one of "gaussian", "binomial", "poisson",
"cox", "multinomial", "mgaussian", or a class "family" object.}

\item{foldid}{Vector of values identifying which fold each observation is
in.}

\item{predict_fun}{The prediction function; see `kfoldcv()` documentation
for details.}

\item{predict_params}{Any other parameters that should be passed tp
`predict_fun` to get predictions (other than `object` and `newx`); see
`kfoldcv()` documentation for details.}

\item{predict_row_params}{A vector which is a subset of
`names(predict_params)`, indicating which parameters have to be subsetted
in the CV loop (other than `newx`); see `kfoldcv()` documentation for
details.}

\item{type.measure}{Loss function to use for cross-validation.
Only required for `family = "cox"`.}

\item{weights}{Observation weights. Only required for `family = "cox"`.}

\item{grouped}{Experimental argument; see `kfoldcv()` documentation for
details. Only required for `family = "cox"`.}
}
\value{
A matrix of predictions.
}
\description{
Build a matrix of predictions from CV model fits.
}
