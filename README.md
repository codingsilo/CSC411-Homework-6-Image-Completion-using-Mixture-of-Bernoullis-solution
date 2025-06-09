# CSC411-Homework-6-Image-Completion-using-Mixture-of-Bernoullis-solution

Download Here: [CSC411 Homework 6: Image Completion using Mixture of Bernoullis solution](https://jarviscodinghub.com/assignment/csc411-homework-6-image-completion-using-mixture-of-bernoullis-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

Submission: You must submit two files through MarkUs1
:
1. a PDF file containing your writeup, titled hw6-writeup.pdf
2. and your completed code file mixture.py.
Neatness Point: One of the 10 points will be given for neatness. You will receive this point as
long as we don’t have a hard time reading your solutions or understanding the structure of your code.
Late Submission: 10% of the marks will be deducted for each day late, up to a maximum of 3
days. After that, no submissions will be accepted.
Collaboration. Weekly homeworks are individual work. Ssee the Course Information handout2
for detailed policies.
Overview
In this assignment, we’ll implement a probabilistic model which we can apply to the task of image
completion. Basically, we observe the top half of an image of a handwritten digit, and we’d like to
predict what’s in the bottom half. An example is shown in Figure 1.
This assignment is meant to give you practice with the E-M algorithm. It’s not as long as it
looks from this handout. The solution requires about 8-10 lines of code.
Mixture of Bernoullis model
The images we’ll work with are all 28 × 28 binary images, i.e. the pixels take values in {0, 1}. We
ignore the spatial structure, so the images are represented as 784-dimensional binary vectors.
A mixture of Bernoullis model is like the other mixture models we’ve discussed in this course.
Each of the mixture components consists of a collection of independent Bernoulli random variables.
I.e., conditioned on the latent variable z = k, each pixel xj is an independent Bernoulli random
variable with parameter θk,j :
p(x
(i)
| z = k) = Y
D
j=1
p(x
(i)
j
| z = k) (1)
=
Y
D
j=1
θ
x
(i)
j
k,j (1 − θk,j )
1−x
(i)
j (2)
Try to understand where this formula comes from. You’ll find it useful when you do the derivations.
1
https://markus.teach.cs.toronto.edu/csc411-2019-01
2
https://www.cs.toronto.edu/~mren/teach/csc411_19s/syllabus.pdf
1
CSC411 Winter 2019 Homework 6
Given these observations… … you want to make these predictions
Figure 1: An example of the observed data (left) and the predictions about the missing part of the
image (right).
This can be written out as the following generative process:
Sample z from a multinomial distribution with parameter vector π.
For j = 1, . . . , D:
Sample xj from a Bernoulli distribution with parameter θk,j , where k is the value of z.
It can also be written mathematically as:
z ∼ Multinomial(π) (3)
xj | z = k ∼ Bernoulli(θk,j ) (4)
Summary of notation
We will refer to three dimensions in our model:
• N = 60,000, the number of training cases. The training cases are indexed by i.
• D = 28 × 28 = 784, the dimension of each observation vector. The dimensions are indexed
by j.
• K, the number of components. The components are indexed by k.
The inputs are represented by X, an N × D binary matrix. In the E-step, we compute R, the
matrix of responsibilities, which is an N × K matrix. Each row gives the responsibilities for one
training case.
2
CSC411 Winter 2019 Homework 6
The trainable parameters of the model, written out as vectors and matrices, are:
π =


π1
π2
.
.
.
πK


Θ =


θ1,1 θ1,2 · · · θ1,D
θ2,1 θ2,2 θ2,D
.
.
.
.
.
.
.
.
.
θK,1 θK,2 · · · θK,D


The rows of Θ correspond to mixture components, and columns correspond to input dimensions.
Part 1: Learning the parameters (3 marks)
In the first step, we’ll learn the parameters of the model given the responsibilities, using the MAP
criterion. This corresponds to the M-step of the E-M algorithm.
In lecture, we discussed the E-M algorithm in the context of maximum likelihood (ML) learning. The MAP case is only slightly different from ML: the only difference is that we add a prior
probability term to the objective function in the M-step. In particular, recall that in the context
of ML, the M-step maximizes the objective function:
X
N
i=1
X
K
k=1
r
(i)
k
h
log Pr(z
(i) = k) + log p(x
(i)
| z
(i) = k)
i
, (5)
where the r
(i)
k
are the responsibilities computed during the E-step. In the MAP formulation, we
add the (log) prior probability of the parameters:
X
N
i=1
X
K
k=1
r
(i)
k
h
log Pr(z
(i) = k) + log p(x
(i)
| z
(i) = k)
i
+ log p(π) + log p(Θ) (6)
Our prior for Θ is as follows: every entry is drawn independently from a beta distribution
with parameters a and b. The beta distribution is discussed in Lecture 14, but here it is again for
reference:
p(θk,j ) ∝ θ
a−1
k,j (1 − θk,j )
b−1
(7)
Recall that ∝ means “proportional to.” I.e., the distribution has a normalizing constant which
we’re ignoring because we don’t need it for the M-step.
For the prior over mixing proportions π, we’ll use the Dirichlet distribution, which is the
conjugate prior for the multinomial distribution. It is a distribution over the probability simplex,
i.e. the set of vectors which define a valid probability distribution.3 The distribution takes the form
p(π) ∝ π
a1−1
1
π
a2−1
2
· · · π
aK−1
K . (8)
For simplicity, we use a symmetric Dirichlet prior where all the ak parameters are assumed to be
equal. Like the beta distribution, the Dirichlet distribution has a normalizing constant which we
3
I.e., they must be nonnegative and sum to 1.
3
CSC411 Winter 2019 Homework 6
don’t need when updating the parameters. The beta distribution is actually the special case of the
Dirichlet distribution for K = 2. You can read more about it on Wikipedia if you’re interested.4
Your tasks for this part are as follows:
1. (2 marks) Derive the M-step update rules for Θ and π by setting the partial derivatives of
Eqn 6 to zero. Your final answers should have the form:
πk ← · · ·
θk,j ← · · ·
Be sure to show your steps. (There’s no trick here; you’ve done very similar questions before.)
2. (1 mark) Take these formulas and use them to implement the functions Model.update_pi
and Model.update_theta in mixture.py. Each one should be implemented in terms of
NumPy matrix and vector operations. Each one requires only a few lines of code, and should
not involve any for loops.
To help you check your solution, we have provided the function checking.check_m_step. If
this check passes, you’re probably in good shape.5
To convince us of the correctness of your implementation, please include the output
of running mixture.print_part_1_values(). Note that we also require you to submit
mixture.py through MarkUs.
3. (0 marks) The function learn_from_labels learns the parameters of the model from the
labeled MNIST images. The values of the latent variables are chosen based on the digit class
labels, i.e. the latent variable z
(i)
is set to k if the ith training case is an example of digit
class k. In terms of the code, this means the matrix R of responsibilities has a 1 in the (i, k)
entry if the ith image is of class k, and 0 otherwise.
Run learn_from_labels to train the model. It will show you the learned components
(i.e. rows of Θ) and print the training and test log-likelihoods. You do not need to submit anything for this part. It is only for your own satisfaction.
Part 2: Posterior inference (3 marks)
Now we derive the posterior probability distribution p(z | xobs), where xobs denotes the subset of the
pixels which are observed. In the implementation, we will represent partial observations in terms
of variables m
(i)
j
, where m
(i)
j = 1 if the jth pixel of the ith image is observed, and 0 otherwise. In
the implementation, we organize the m
(i)
j
’s into a matrix M which is the same size as X.
1. (1 mark) Derive the rule for computing the posterior probability distribution p(z | x). Your
final answer should look something like
Pr(z = k | x) = · · · (9)
where the ellipsis represents something you could actually implement. Note that the image
may be only partially observed.
4
https://en.wikipedia.org/wiki/Dirichlet_distribution
5
It’s worth taking a minute to think about why this check works. It’s based on the variational interpretation of
E-M discussed at the end of Lecture 17. You can also read more about it in Neal and Hinton, 1998, “A view of the
E-M algorithm that justifies incremental, sparse, and other variants.”
4
CSC411 Winter 2019 Homework 6
Hints: For this derivation, you probably want to express the observation probabilities in the
form of Eqn 2.
2. (1 mark) Implement the method Model.compute_posterior using your solution to the previous question. While your answer to Question 1 was probably given in terms of probabilities,
we do the computations in terms of log probabilities for numerical stability. We’ve already
filled in part of the implementation, so your job is to compute log p(z, x) as described in the
method’s doc string.
Your implementation should use NumPy matrix and vector operations, rather than a for
loop. Hint: There are two lines in Model.log_likelihood which are almost a solution to
this question. You can reuse these lines as part of the solution, except you’ll need to modify
them to deal with partial observations.
To help you check your solution, we’ve provided the function checking.check_e_step. Note
that this check only covers the case where the image is fully observed, so it doesn’t fully verify
your solution to this part.
3. (1 mark) Implement the method Model.posterior_predictive_means, which computes the
posterior predictive means of the missing pixels given the observed ones. Hint: this requires
only two very short lines of code, one of which is a call to Model.compute_posterior.
To convince us of the correctness of the implementation for this part and the previous part,
please include the output of running mixture.print_part_2_values(). Note that we
also require you to submit mixture.py through MarkUs.
4. (0 marks) Run the function train_with_em, which trains the mixture model using E-M.
It plots the log-likelihood as a function of the number of steps.6 You can watch how the
mixture components change during training.7
It also shows the model’s image completions
after every step. You can watch how they improve over the course of training. At the very
end, it outputs the training and test log-likelihoods. The final model for this part should be
much better than the one from Part 1. You do not need to submit anything for this part. It’s
only for your own satisfaction.
Part 3: Conceptual questions (3 marks)
This section asks you to reflect on the learned model. We tell you the outcomes of the experiments,
so that you can do this part independently of the first 2. Each question can be answered in
a few sentences.
1. (1 mark) In the code, the default parameters for the beta prior over Θ were a = b = 2. If
we instead used a = b = 1 (which corresponds to a uniform distribution), the MAP learning
algorithm would have the problem that it might assign zero probability to images in the test
set. Why might this happen? Hint: what happens if a pixel is always 0 in the training set,
but 1 in the test image?
6Observe that it uses a log scale for the number of E-M steps. This is always a good idea, since it can be difficult
to tell if the training has leveled off using a linear scale. You wouldn’t know if it’s stopped improving or is just
improving very slowly.
7
It’s likely that 5-10 of the mixture components will “die out” during training. In general, this is something we
would try to avoid using better initializations and/or priors, but in the context of this assignment it’s the normal
behavior.
5
CSC411 Winter 2019 Homework 6
2. (1 mark) The model from Part 2 gets significantly higher average log probabilities on both
the training and test sets, compared with the model from Part 1. This is counterintuitive,
since the Part 1 model has access to additional information: labels which are part of a true
causal explanation of the data (i.e. what digit someone was trying to write). Why do you
think the Part 2 model still does better?
3. (1 mark) The function print_log_probs_by_digit_class computes the average log-probabilities
for different digit classes in both the training and test sets. In both cases, images of 1’s are
assigned far higher log-probability than images of 8’s. Does this mean the model thinks 1’s
are far more common than 8’s? I.e., if you sample from its distribution, will it generate far
more 1’s than 8’s? Why or why not?
