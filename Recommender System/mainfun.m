%% Loading movie ratings dataset
load ('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i


%% Entering ratings for a new user
%  Before we will train the collaborative filtering model, we will first
%  add ratings that correspond to a new user that we just observed. 
movieList = loadMovieList();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

my_ratings(1) = 4;
my_ratings(98) = 2;
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

%% Learning Movie Ratings
%  Now, you will train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users

fprintf('\nTraining collaborative filtering...\n');

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters 
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set lambda for Regularization
lambda = 10;
sol = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies,num_features, lambda)),initial_parameters, options);

% Unfold the returned sol back into U and W
X = reshape(sol(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(sol(num_movies*num_features+1:end), num_users, num_features);

fprintf('Recommender system learning completed.\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Recommendation for new user
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.

p = X * Theta';
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j),movieList{j});
end

fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i),movieList{i});
    end
end
