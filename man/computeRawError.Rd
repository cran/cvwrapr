% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/computeRawError.R
\name{computeRawError}
\alias{computeRawError}
\title{Compute the nobs by nlambda matrix of errors}
\usage{
computeRawError(predmat, y, type.measure, family, weights, foldid, grouped)
}
\arguments{
\item{predmat}{Array of predictions. If `y` is univariate, this has
dimensions `c(nobs, nlambda)`. If `y` is multivariate with `nc`
levels/columns (e.g. for `family = "multionmial"` or
`family = "mgaussian"`), this has dimensions `c(nobs, nc, nlambda)`.
Note that these should be on the same scale as `y` (unlike in the
glmnet package where it is the linear predictor).}

\item{y}{Response variable.}

\item{type.measure}{Loss function to use for cross-validation. See
`availableTypeMeasures()` for possible values for `type.measure`. Note that
the package does not check if the user-specified measure is appropriate
for the family.}

\item{family}{Model family; used to determine the correct loss function.}

\item{weights}{Observation weights.}

\item{foldid}{Vector of values identifying which fold each observation is
in.}

\item{grouped}{Experimental argument; see `kfoldcv()` documentation for
details.}
}
\value{
A list with the following elements:
\item{cvraw}{An nobs by nlambda matrix of raw error values.}
\item{weights}{Observation weights.}
\item{N}{A vector of length nlambda representing the number of non-NA
predictions associated with each lambda value.}
\item{type.measure}{Loss function used for CV.}
}
\description{
Computes the nobs by nlambda matrix of errors corresponding to the error
measure provided. Only works for "gaussian" and "poisson" families right
now.
}
