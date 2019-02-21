# Oil-field-performance
#Random forest codes for estimating recovery factor, maximum oil rate and average depletion rate
#Load and Pre-process Data
load data

#The code below will convert loaded numerical variables that are categorical into categorical variable.
Diagenesis = categorical(Diagenesis);
Paleoclimate = categorical(Paleoclimate);
GrossDepEnvironment = categorical(GrossDepEnvironment);
ProductionStrategy = categorical(ProductionStrategy);
StratigraphicHeterogeneity = categorical(StratigraphicHeterogeneity);
TrapType = categorical(Traptype);
StructuralComplexity = categorical(StructuralComplexity);

#Make a table X containing all the training data using 32 variables
X = table(GDE,Depth,averageporosity,averagepermeability,initial pressure,structuralcomplexity,Netgrossratio...
    Averagedepletionrate,RecoveryFactor);

#Determine the number of Levels in Predictors
countLevels = @(x)numel(categories(categorical(x)));
numLevels = varfun(countLevels,X(:,1:end-1),'OutputFormat','uniform');

#Grow Robust Random Forest
#For the three models we had Mdl1, Mdl2 and Mdl3, with X, Y and Z tables corresponding to recovery factor, maximum reservoir rate and depletion rate models respectively.
t = templateTree('NumVariablesToSample','all',...
    'PredictorSelection','interaction-curvature','Surrogate','on');
Mdl = fitrensemble(X,'RecoveryFactor','Method','bag','NumLearningCycles',500,...
    'Learners',t);

#Estimate the model R2 using out-of-bag predictions.
yHat = oobPredict(Mdl);
R2 = corr(Mdl.Y,yHat)^2

#Predictor Importance Estimation
impOOB1 = oobPermutedPredictorImportance(Mdl1);
impOOB2 = oobPermutedPredictorImportance(Mdl2);
impOOB3 = oobPermutedPredictorImportance(Mdl3);

#Make prediction on new data
# V below is the test dataset
YHat = predict(Mdl,X)
