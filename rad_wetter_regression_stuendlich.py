# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:08:11 2021

@author: katha
"""
# --------------------------------------------------------------------
# change directory
# --------------------------------------------------------------------
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import holidays
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
from additional_functions import *
from scipy import stats
import seaborn as sns
from patsy import dmatrices
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(r"C:\Users\katha\OneDrive\Dokumente\GitHub\Fahrrad_Muenchen_2020")

# --------------------------------------------------------------------
# import modules
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# load and merge data
# --------------------------------------------------------------------
df_weather = pd.read_pickle('weather')
df_fahrrad = pd.read_pickle('radzaehldaten_stunde')
df_radwetter = df_weather.set_index('Zeit').join(
    df_fahrrad.set_index('Zeit'), how='inner').reset_index()

# --------------------------------------------------------------------
# drop Kreuther
# --------------------------------------------------------------------
df_radwetter = df_radwetter[df_radwetter['zaehlstelle'] != 'Kreuther']
df_radwetter = df_radwetter.reset_index(drop=True)

# --------------------------------------------------------------------
# get day information
# --------------------------------------------------------------------
# Holiday
df_radwetter['Feiertag'] = ""
for i in range(len(df_radwetter)):
    df_radwetter.Feiertag[i] = df_radwetter.Zeit[i] in holidays.Germany(
        prov='BY')

# weekday
df_radwetter['Wochentag'] = ""
df_radwetter['Wochenende'] = ""
for i in range(len(df_radwetter)):
    day = df_radwetter.Zeit[i].weekday()
    df_radwetter.Wochentag[i] = day
    df_radwetter.Wochenende[i] = day > 4

df_radwetter['Werktag'] = df_radwetter.Wochenende + df_radwetter.Feiertag
df_radwetter.Werktag[df_radwetter.Werktag > 0] = 1
df_radwetter.Werktag = (df_radwetter.Werktag - 1) * (-1)


# season
def get_meteo_season(date):
    if date.month in [12, 1, 2]:
        season = 'winter'
    elif date.month in [3, 4, 5]:
        season = 'spring'
    elif date.month in [6, 7, 8]:
        season = 'summer'
    elif date.month in [9, 10, 11]:
        season = 'fall'
    return season


df_radwetter['Jahreszeit'] = ""
for i in range(len(df_radwetter)):
    df_radwetter.loc[i, ['Jahreszeit']] = get_meteo_season(
        df_radwetter.Zeit[i])

# Monat
df_radwetter['Monat'] = ""
for i in range(len(df_radwetter)):
    df_radwetter.loc[i, ['Monat']] = df_radwetter['Zeit'][i].month

df_radwetter['Tageslicht'] = df_radwetter['Solarstrahlung'] > 0


df_radwetter.to_pickle('radwetter')


####################### start here ######################
df_radwetter = pd.read_pickle('radwetter')

# --------------------------------------------------------------------
# Corona?
# --------------------------------------------------------------------
df_radwetter.Corona = []
for i,row in df_radwetter.iterrows():
    if (row['Zeit'].year == 2020) & (row['Zeit'].month >= 3):
        df_radwetter.loc[i,'Corona'] = True
    elif  (row['Zeit'].year == 2021): 
        df_radwetter.loc[i,'Corona'] = True
    else:
        df_radwetter.loc[i,'Corona'] = False

df_radwetter = df_radwetter.loc[df_radwetter.Corona == 0]

# --------------------------------------------------------------------
# eliminate zaehlstelle
# --------------------------------------------------------------------
df_radwetter['gesamt_all'] = df_radwetter.groupby(
    ['Zeit'])['gesamt'].transform('sum')
del df_radwetter['zaehlstelle']  # del df_weather['Solarstrahl.']
del df_radwetter['gesamt']  # del df_weather['Solarstrahl.']
df_radwetter = df_radwetter.drop_duplicates().reset_index(drop=True)

# ----------------
# transform
# --------------------------

# Commute = Gesamt_Werktag_per_Hour - Median_Gesamt_NichtWerktag_Per_Hour
# rush_Hour: more than average per hour (>213)
zaehlung_feiertag = df_radwetter[df_radwetter.Werktag == 0].groupby(
    ['Hour'])['gesamt_all'].median().reset_index(name='zaehlung')
df_radwetter_werktag = df_radwetter[df_radwetter.Werktag == 1].reset_index()
df_radwetter_werktag['gesamt_communte'] = ""
for i in range(len(df_radwetter_werktag)):
    n_feiertag = zaehlung_feiertag.loc[zaehlung_feiertag.Hour ==
                                       df_radwetter_werktag.loc[i, 'Hour']].zaehlung.values[0]
    if (df_radwetter_werktag.loc[i, 'gesamt_all'] - n_feiertag) < 0:
        df_radwetter_werktag.loc[i, 'gesamt_communte'] = float(0)
    else:
        df_radwetter_werktag.loc[i, 'gesamt_communte'] = float(
            df_radwetter_werktag.loc[i, 'gesamt_all'] - n_feiertag)


RushHour_selection = df_radwetter_werktag.groupby(
    ['Hour'])['gesamt_communte'].apply(np.median) > 200
RushHour = RushHour_selection.index[RushHour_selection]
df_radwetter_rushhour = df_radwetter_werktag[df_radwetter_werktag['Hour'].isin(
    RushHour)].reset_index(drop=True)
df_radwetter_rushhour_morning = df_radwetter_werktag[df_radwetter_werktag['Hour'].isin(
    [7, 8, 9])].reset_index(drop=True)
df_radwetter_rushhour_evening = df_radwetter_werktag[df_radwetter_werktag['Hour'].isin(
    [17, 18, 19, 20])].reset_index(drop=True)

# Regen: Binary

df_radwetter_rushhour_morning['Regen_binary'] = ""
df_radwetter_rushhour_morning.loc[df_radwetter_rushhour_morning["Regen"]
                                  == 0, 'Regen_binary'] = 0
df_radwetter_rushhour_morning.loc[df_radwetter_rushhour_morning["Regen"]
                                  > 0, 'Regen_binary'] = 1

df_radwetter_rushhour_evening['Regen_binary'] = ""
df_radwetter_rushhour_evening.loc[df_radwetter_rushhour_evening["Regen"]
                                  == 0, 'Regen_binary'] = 0
df_radwetter_rushhour_evening.loc[df_radwetter_rushhour_evening["Regen"]
                                  > 0, 'Regen_binary'] = 1

df_radwetter_rushhour['Regen_binary'] = ""
df_radwetter_rushhour.loc[df_radwetter_rushhour["Regen"]
                          == 0, 'Regen_binary'] = 0
df_radwetter_rushhour.loc[df_radwetter_rushhour["Regen"]
                          > 0, 'Regen_binary'] = 1


# ----------------
# standardize
# --------------------------
# Tempertaur
df_radwetter_rushhour_morning['zTemperatur'] = (df_radwetter_rushhour_morning.Temperatur - np.mean(
    df_radwetter_rushhour_morning.Temperatur))/np.std(df_radwetter_rushhour_morning.Temperatur)
df_radwetter_rushhour_evening['zTemperatur'] = (df_radwetter_rushhour_evening.Temperatur - np.mean(
    df_radwetter_rushhour_evening.Temperatur))/np.std(df_radwetter_rushhour_evening.Temperatur)
df_radwetter_rushhour['zTemperatur'] = (df_radwetter_rushhour.Temperatur - np.mean(
    df_radwetter_rushhour.Temperatur))/np.std(df_radwetter_rushhour.Temperatur)

# --------------------------------------------------------------------
# plot timecourse
# --------------------------------------------------------------------
#df_radwetter[['Zeit', 'Temperatur', 'Regen']].set_index('Zeit').plot()

df_radwetter.groupby(['Hour', 'Werktag'])[
    'gesamt_all'].median().unstack().plot()
df_radwetter_werktag.groupby(['Hour'])['gesamt_communte'].apply(np.median).plot()
plt.ylabel('Fahrradaufkommen')
plt.axhline(y=213, color='grey', linestyle='--')
plt.axvline(x=6.5, color='grey', linestyle='--')
plt.axvline(x=9.7, color='grey', linestyle='--')
plt.axvline(x=16.5, color='grey', linestyle='--')
plt.axvline(x=20.5, color='grey', linestyle='--')
# --------------------------------------------------------------------
# plot count data
# --------------------------------------------------------------------
sns.distplot((df_radwetter_rushhour_morning.gesamt_all)**(1/2))
sns.distplot((df_radwetter_rushhour_morning.gesamt_communte)**(1/2))
sns.distplot(df_radwetter_rushhour_evening.gesamt_communte)

# --------------------------------------------------------------------
# check normality
# --------------------------------------------------------------------
k2, p = stats.normaltest(
    np.log(df_radwetter_rushhour_morning.gesamt_communte))
k2, p = stats.normaltest((df_radwetter_rushhour_morning.gesamt_all)**(1/2))

df_radwetter_rushhour_morning['gesamt_communte_sqrt'] = df_radwetter_rushhour_morning.gesamt_communte**(
    1/2)
df_radwetter_rushhour_evening['gesamt_communte_sqrt'] = df_radwetter_rushhour_evening.gesamt_communte**(
    1/2)
df_radwetter_rushhour['gesamt_communte_sqrt'] = df_radwetter_rushhour.gesamt_communte**(
    1/2)
# --------------------------------------------------------------------
# First visual inspection of association with Fahrradaufkommen
# --------------------------------------------------------------------
sns.boxplot(x="Tageslicht", y="gesamt_communte_sqrt",
            data=train, color = 'blue')
plt.xlabel("Tageslicht")
plt.ylabel("Fahrradaufkommen")
plt.show()

# # Niederschlag (binary or linear?)
sns.boxplot((train.Regen_binary),
            train.gesamt_communte,  color='blue')
plt.xlabel("Niederschlag")
plt.ylabel("Fahrradaufkommen")
plt.show()


# # Temperatur linear of curvilinear?
plt.scatter(df_radwetter_rushhour_morning.Temperatur,
            df_radwetter_rushhour_morning.gesamt_communte_sqrt,  color='blue')
plt.xlabel("Temperatur")
plt.ylabel("Fahrradaufkommen")
plt.show()

# sns.scatterplot(data=df_radwetter_daytime, x="Temperatur", y="gesamt", hue = 'Wochenende')


# # Sonnenstunden
sns.scatterplot((train.Sonnenschein),
            train.gesamt_communte,  color='blue')
plt.xlabel("sonnenstunden")
plt.ylabel("Fahrradaufkommen")
plt.show()
# ax = sns.boxplot(x="csonnenstunden", y="gesamt3", data=df_radwetter)


sns.scatterplot((train.Windgeschwindigkeit),
            train.gesamt_communte_sqrt,  color='blue')
plt.xlabel("Windgeschwindigkeit")
plt.ylabel("Fahrradaufkommen")
plt.show()

# --------------------------------------------------------------------
# split data
# --------------------------------------------------------------------

regressors = []
###
for x in range(1000):  
    msk = np.random.rand(len(df_radwetter_rushhour)) < 0.5
    train = df_radwetter_rushhour[msk].reset_index(drop=True)
    test = df_radwetter_rushhour[~msk].reset_index(drop=True)
    

# data inspection
#df_count_obs = pd.DataFrame(columns=train.columns,  index = range(len(train)))

# for i in range(len(train)):
#    df_count_obs.Regen[i] = math.ceil(train.Regen[i])
#    df_count_obs.Temperatur[i] = math.ceil(train.Temperatur[i] /5)
#    df_count_obs.Sonnenschein[i] = math.ceil(train.Sonnenschein[i] *3)
#    df_count_obs.Windgeschwindigkeit[i] =  math.ceil(train.Windgeschwindigkeit[i] /4)

#df_count_obs.Werktag= train.Werktag
#counts = df_count_obs.groupby(['Temperatur','Regen','Sonnenschein', 'Windgeschwindigkeit']).size().reset_index(name='counts')
# print(set(counts.counts))
#n_obs = counts.counts.value_counts()

# --------------------------------------------------------------------
# Multicollinearity
# --------------------------------------------------------------------
    # rtrain = train[['gesamt_communte_sqrt', 'zTemperatur', 'Regen',
    #                 'Sonnenschein', 'Windgeschwindigkeit', 'Jahreszeit', 'Tageslicht']]
    # #rtrain['Werktag'] = rtrain['Werktag'].astype(int).astype(str)
    
    # features = "+".join(rtrain.columns[1:])
    # y = rtrain.columns[0]
    # y_train, X_train = dmatrices(
    #     y + '~' + features, train, return_type='dataframe')
    # # multicollinearity
    # vif = pd.DataFrame()
    # vif["VIF Factor"] = [variance_inflation_factor(
    #     X_train.values, i) for i in range(X_train.shape[1])]
    # vif["features"] = X_train.columns
    # vif.round(1)

# --------------------------------------------------------------------
# Regression: Prepare
# --------------------------------------------------------------------

    # # get variables of interest
    # rtrain = train[['gesamt_communte_sqrt', 'Regen', 'Regen_binary', 'Temperatur']]
    # #rtrain['Werktag'] = rtrain['Werktag'].astype(int).astype(str)
    # y = rtrain.columns[0]
    # rtrain['gesamt_communte_sqrt'] = pd.Categorical(rtrain['gesamt_communte_sqrt'])
    # rtrain[['gesamt_communte_sqrt']] = rtrain['gesamt_communte_sqrt'].cat.codes
    # min_max_scaler = MinMaxScaler()
    # rtrain['gesamt_communte_sqrt'] = min_max_scaler.fit_transform(
    #     rtrain[['gesamt_communte_sqrt']])
    
    # --------------------------------------------------------------------
    # Regression: Choose Regressors
    # --------------------------------------------------------------------
    # # Regen
    # model_regen_cont = ols(y + '~ Regen', data=rtrain).fit()  # 0.029
    # model_regen_binary = ols(y + '~ Regen_binary', data=rtrain).fit()  # 0.068
    # print(model_regen_cont.rsquared_adj)
    # print(model_regen_binary.rsquared_adj)
    
    
    # # Temperatur
    # mymodel = np.poly1d(np.polyfit(rtrain.Temperatur,
    #                     rtrain.gesamt_communte_sqrt, 1))
    # print(r2_score(rtrain.gesamt_communte_sqrt, mymodel(rtrain.Temperatur)))
    
    # polynomial_features = PolynomialFeatures(degree=4)
    # temp = polynomial_features.fit_transform(
    #     np.array(rtrain.Temperatur).reshape((len(rtrain.Temperatur), 1)))
    
    # model_poly = sm.OLS(rtrain.gesamt_communte_sqrt, temp).fit()
    # ypred = model_poly.predict(temp)
    # #plt.scatter(rtrain.Temperatur, rtrain.gesamt_communte_sqrt)
    # plt.plot(rtrain.Temperatur, ypred, '.')
    # print(model_poly.rsquared_adj)
    # print(model_poly.bic)
    
    # --------------------------------------------------------------------
    # Regression: Main effect
    # --------------------------------------------------------------------
    rtrain = train[['gesamt_communte_sqrt', 'Temperatur', 'Regen_binary',
                    'Sonnenschein', 'Windgeschwindigkeit', 'Tageslicht']]
    y = rtrain.columns[0]
    rtrain['gesamt_communte_sqrt'] = pd.Categorical(rtrain['gesamt_communte_sqrt'])
    rtrain[['gesamt_communte_sqrt']] = rtrain['gesamt_communte_sqrt'].cat.codes
    min_max_scaler = MinMaxScaler()
    rtrain['gesamt_communte_sqrt'] = min_max_scaler.fit_transform(
        rtrain[['gesamt_communte_sqrt']])
    
    features = "+".join(rtrain.columns[1:])
    main_effect_model = ols(y + '~' + features, data=rtrain).fit()
    main_effect_model.summary()
    
    
    print(main_effect_model.summary())
    # kick out Windgeschwindigkeit?
    names_0 = dict(main_effect_model.params)
    names = make_variable_list(list(names_0.keys()))
    # names.remove('Windgeschwindigkeit')
    
    # --------------------------------------------------------------------
    # Model Fitting: Interaction Effects
    # --------------------------------------------------------------------
    # prepare model fitting
    i = 0
    iterations_log = ""
    # 1 fit initial full model
    new_formula = y + ' ~' + '*'.join(names)
    interaction_model_full = ols(new_formula, data=rtrain).fit()
        # if we want to remove higher order interactions
    names_0 = dict(interaction_model_full.params)
    names = make_variable_list(list(names_0.keys()))
    names = remove_higher_order_interactions(names)  # get rid of 3was interactions
    try:
        names.remove('Sonnenschein:Tageslicht')
    except:
        names.remove('Tageslicht:Sonnenschein')
    new_formula = y + ' ~' + '+'.join(names)
    #y_train, X_train = dmatrices(new_formula, train, return_type='dataframe')
    #best_model_so_far = smf.glm(formula = new_formula, data=train, family=sm.families.NegativeBinomial()).fit()
    #sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    best_model_so_far = ols(new_formula, data=rtrain).fit()
    
    print(best_model_so_far.summary())
    # 1-(best_model_so_far.deviance/best_model_so_far.null_deviance)
    
    # 2 fit model with worst interaction removed
    # select worst interaction effect
    dict_pvalues = dict(best_model_so_far.pvalues[1:])
    dict_pvalues_interactions = {i: dict_pvalues[i]
                                 for i in dict_pvalues.keys() if ':' in i}
    # only keep worst interaction for categorical
    dict_pvalues_interactions_new = dict()
    for k in dict_pvalues_interactions.keys():
        if not remove_brackets(k) in dict_pvalues_interactions_new:
            dict_pvalues_interactions_new[remove_brackets(
                k)] = dict_pvalues_interactions[k]
        else:
            if dict_pvalues_interactions_new[remove_brackets(k)] > dict_pvalues_interactions[k]:
                dict_pvalues_interactions_new[remove_brackets(
                    k)] = dict_pvalues_interactions[k]
    maxPval = max(dict_pvalues_interactions_new.values())
    worst_param = max(dict_pvalues_interactions_new,
                      key=dict_pvalues_interactions_new.get)
    names.remove(worst_param)
    iterations_log += "\n ####### \n" + str(i) + "AIC: " + str(
        best_model_so_far.aic) + "\nworst p: " + str(maxPval) + "\nworst B: " + str(worst_param)
    print("Character Variables (Dropped):" + str(worst_param))
    
    
    # 3 fit next best model: worst parameter removed
    i += 1
    new_formula = y + ' ~' + '+'.join(names)
    next_best_model = ols(new_formula, data=rtrain).fit()
    #next_best_model = smf.glm(formula = new_formula, data=train, family=sm.families.NegativeBinomial()).fit()
    
    #y_train, X_train = dmatrices(new_formula, train, return_type='dataframe')
    #next_best_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(next_best_model.summary())
    
    # compare models: is it better without worst parameter?
    while (next_best_model.bic < best_model_so_far.bic) and len(dict_pvalues_interactions) > 0:
        iterations_log += '\n' + 'Dropped ' + str(worst_param)
        best_model_so_far = next_best_model
        try:
            # remove worst interaction for next model
            dict_pvalues = dict(best_model_so_far.pvalues[1:])
            dict_pvalues_interactions = {
                i: dict_pvalues[i] for i in dict_pvalues.keys() if ':' in i}
            dict_pvalues_interactions_new = dict()
            for k in dict_pvalues_interactions.keys():
                if not remove_brackets(k) in dict_pvalues_interactions_new:
                    dict_pvalues_interactions_new[remove_brackets(
                        k)] = dict_pvalues_interactions[k]
                else:
                    if dict_pvalues_interactions_new[remove_brackets(k)] > dict_pvalues_interactions[k]:
                        dict_pvalues_interactions_new[remove_brackets(
                            k)] = dict_pvalues_interactions[k]
            maxPval = max(dict_pvalues_interactions_new.values())
            worst_param = max(dict_pvalues_interactions_new,
                              key=dict_pvalues_interactions_new.get)
            names.remove(worst_param)
            print("Character Variables (Dropped):" + str(worst_param))
            # 4 fit next model
            i += 1
            #features = "+".join(names)
            #y_train, X_train = dmatrices(y + '~' + features, train, return_type='dataframe')
            #next_best_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
            new_formula = y + ' ~' + '+'.join(names)
            next_best_model = ols(new_formula, data=rtrain).fit()
            # smf.glm(formula = new_formula, data=train, family=sm.families.NegativeBinomial()).fit()
        except:
            print('No Interaction effects left')
    final_model = best_model_so_far
    print(final_model.summary())
    print(iterations_log)
    
    #ypred = final_model.predict(rtrain[1:])
    
    # --------------------------------------------------------------------
    # Effect Size: standardized Beta
    # --------------------------------------------------------------------
    # make X matrix to estimate std later
    df_train_interaction = pd.DataFrame(index = range(len(rtrain)))
    
    names_0 = dict(final_model.params)
    names = make_variable_list(list(names_0.keys()))
    for p in names:
        if ':' not in p:
            df_train_interaction[p] = rtrain[p]
        else:
            ps = p.split(':')
            df_train_interaction[p] = rtrain[ps[0]]*rtrain[ps[1]]
    
    # make dictionary of standardized beta's
    p_dict = dict(final_model.params)
    del p_dict['Intercept']
    zB_dict = dict()
    for i, p in enumerate(list(p_dict.keys())):
        if p != 'Intercept':
            zBeta = p_dict[p] * np.std(df_train_interaction[remove_brackets(p)])/np.std(rtrain.gesamt_communte_sqrt)
        zB_dict[remove_brackets(p)] = zBeta
    regressors.extend(sorted(list(zB_dict.keys())))

# ----------------------------------------------
# ist das model konsistent?
# ----------------------------------------------

final_regressors = []
for r in list(set(regressors)):
    if regressors.count(r) > 800:
        final_regressors.append(r)
# ----------------------------------------------
# das finale model
# ----------------------------------------------
msk = np.random.rand(len(df_radwetter_rushhour)) < 0.5
train = df_radwetter_rushhour[msk].reset_index(drop=True)
test = df_radwetter_rushhour[~msk].reset_index(drop=True)
    # # get variables of interest
y = 'gesamt_communte_sqrt'
train['gesamt_communte_sqrt'] = pd.Categorical(train['gesamt_communte_sqrt'])
train[['gesamt_communte_sqrt']] = train['gesamt_communte_sqrt'].cat.codes
min_max_scaler = MinMaxScaler()
train['gesamt_communte_sqrt'] = min_max_scaler.fit_transform(train[['gesamt_communte_sqrt']])

   
final_formula = y + ' ~' + '+'.join(final_regressors)
final_model = ols(final_formula, data=train).fit()
print(final_model.summary())
  
# ----------------------------------------------
# effect size of final model  
# ----------------------------------------------
df_train_interaction = pd.DataFrame(index = range(len(train)))
names_0 = dict(final_model.params)
names = make_variable_list(list(names_0.keys()))
for p in names:
    if ':' not in p:
        df_train_interaction[p] = train[p]
    else:
        ps = p.split(':')
        df_train_interaction[p] = train[ps[0]]*train[ps[1]]

# make dictionary of standardized beta's
p_dict = dict(final_model.params)
del p_dict['Intercept']
zB_dict = dict()
for i, p in enumerate(list(p_dict.keys())):
    if p != 'Intercept':
        zBeta = p_dict[p] * np.std(df_train_interaction[remove_brackets(p)])/np.std(train.gesamt_communte_sqrt)
    zB_dict[remove_brackets(p)] = zBeta


# ###############

# test_x = np.asanyarray(test[['ENGINESIZE']])
# test_y = np.asanyarray(test[['CO2EMISSIONS']])
# score = final_model.predict(test)
# print("R2-score: %.2f" % r2_score(test[['gesamt3']] , score) )
# #from sklearn.metrics import r2_score

# #test_x = np.asanyarray(test[['ENGINESIZE']])
# #test_y = np.asanyarray(test[['CO2EMISSIONS']])
# #score = final_model.predict(test)
# #print("R2-score: %.2f" % r2_score(test[['gesamt3']] , score) )

# --------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------
from sklearn.model_selection import cross_val_score
y = train.columns[0]
X_train_const = []
for i, row in df_radwetter_rushhour.iterrows():
    temp_list = [1]
    for p in names:
        if ':' not in p:
            temp_list.append(row[p])
        else:
            ps = p.split(':')
            temp_list.append(row[ps[0]]*row[ps[1]])
    X_train_const.append(temp_list)
df_radwetter_rushhour['gesamt_communte_sqrt'] = pd.Categorical(df_radwetter_rushhour['gesamt_communte_sqrt'])
df_radwetter_rushhour[['gesamt_communte_sqrt']] = df_radwetter_rushhour['gesamt_communte_sqrt'].cat.codes
min_max_scaler = MinMaxScaler()
df_radwetter_rushhour['gesamt_communte_sqrt'] = min_max_scaler.fit_transform(
    df_radwetter_rushhour[['gesamt_communte_sqrt']])

clf  = ols(final_formula, data=df_radwetter_rushhour).fit()
clf = LinearRegression()
scores  = cross_val_score(clf,X_train_const,df_radwetter_rushhour.gesamt_communte_sqrt, scoring = 'explained_variance', cv = 2)
score = final_model.predict(test)
print("R2-score: %.2f" % r2_score(test.gesamt_communte_sqrt , score))

# --------------------------------------------------------------------
# Interpretation of COefficients
# --------------------------------------------------------------------

Regen
  -0.0669 +  *np.mean(train.Temperatur)

Temperatur
0.0139 + -0.0075 
  


    

