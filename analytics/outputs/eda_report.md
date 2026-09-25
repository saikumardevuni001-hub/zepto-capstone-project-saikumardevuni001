# EDA results

## IQR outliers
- age: 65
- fare: 114

## Fare distribution
- mean: 32.0967
- median: 14.4542
- mode: 8.0500
- conclusion: right-skewed

## Survival rate by sex
sex
female    74.038462
male      18.890815

## Survival rate by pclass
pclass
1    62.616822
2    47.282609
3    24.236253

## Survival rate by sex and pclass
sex     pclass
female  1         96.739130
        2         92.105263
        3         50.000000
male    1         36.885246
        2         15.740741
        3         13.544669

## Two strongest absolute off-diagonal correlations
1. pclass vs fare: -0.5482
2. sibsp vs parch: 0.4145

## Chart interpretations
### Chart 1
Women show higher survival proportions than men, while first-class groups generally show higher survival than lower classes. The grouped view makes the interaction between sex and passenger class visible rather than treating either variable in isolation.

### Chart 2
Survivors have a higher fare distribution than non-survivors in this dataset. This is consistent with fare acting partly as a proxy for socioeconomic status and passenger class.

### Chart 3
The scatter shows survival observations distributed across age and fare rather than being determined by one variable alone. Higher-fare observations are more concentrated in groups with higher survival proportions.

### Chart 4
Passenger counts and survival counts vary substantially by class. The plot complements the rate-based analysis by showing the underlying group sizes.