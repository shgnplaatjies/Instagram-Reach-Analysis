# Instagram-Reach-Analysis
Instagram reach analysis using python and tensorflow.
This actually turned into a really interesting pipelining problem.

It is very useful to be able to be able to visualize code in different ways, but it is  very tricky and takes a lot of technical know-how and often the ability to write and interpret code.

I am working on a solution to this problem by generalizing the plotly API function calls to one single function.

There are 13 Categories of data visualization plots on plotly:
-- Basic
-- Part-of-whole
-- 1D Distributions
-- 2D Distributions
-- 3 Dimensional
-- Matrix
-- Multi Dimensional
-- Tile Maps
-- Outline Maps
--  Polar Charts
-- Ternary Charts
-- Text Charts

Each with various specific types of plots in each category, this codebase previously contained swathes of boilerplate necessary to generalize to 40 of these functions to improve it's user friendliness, like scatter 3d, scatter, scatter_matrix, scatter_geo, etc.

To reduce boring typing and boilerplate code. I've used a software design principle, higher-order functions, to dynamically call different plotly methods generalize this method.

Each function has a different set of compulsory and optional parameters, this forms a large number of potential combinations an end-user might configure at runtime. I.e., some functions are incompatible with the size parameter, but it's optional for almost all types of plot. The generalized plotFeatureVsTarget method must  account for this)

It should be okay to interrupt the user experience with an exception when they include, within the dynamic function call, a parameter which is either of the wrong type (which should be very difficult...) or it's included and incompatible. (TBH... There should be a way to try/catch/except for that, i.e., to just ignore their size parameter if it's incompatible with a line graph)
