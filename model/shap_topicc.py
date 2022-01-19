# This is meant to be run in Jupyter in combination with basic.py or full.py
# It assumes an X and y and various models

import shap

#background = shap.maskers.Independent(test_features,max_samples=100)
#explainer = shap.Explainer(rf,background)

language_map = {'LowpH': 'Lowest pH',
    'HighpH': 'Highest pH',
    'HighPCO2': 'Highest PCO2',
    'LowPa02': 'Lowest PaO2',
    'LowIonCalcium': 'Lowest iCa',
    'HighIonCalcium': 'Highest iCa',
    'LowSodium': 'Lowest Sodium',
    'HighSodium': 'Highest Sodium',
    'HighPotassium': 'Highest Potassium',
    'HighBUN': 'Highest BUN',
    'HighCreatinine': 'Highest Creatinine',
    'LowGlucose': 'Lowest Glucose',
    'HighGlucose': 'Highest Glucose',
    'LowCO2': 'Lowest Bicarb',
    'HighCO2': 'Highest Bicarb',
    'LowTotalCalcium': 'Lowest Calcium',
    'HighTotalCalcium': 'Highest Calcium',
    'LowWBC': 'Lowest WBC',
    'HighWBC': 'Highest WBC',
    'LowHemoglobin': 'Lowest Hgb',
    'HighHemoglobin': 'Highest Hgb',
    'LowPlatelets': 'Lowest PLT',
    'HighPT': 'Highest PT',
    'HighPTT': 'Highest PTT',
    'HighINR': 'Highest INR',
    'HighTemp': 'Highest Temp',
    'LowTemp': 'Lowest Temp',
    'HighRespRate': 'Highest RR',
    'LowRespRate': 'Lowest RR',
    'HighHeartRate': 'Highest HR',
    'LowHeartRate': 'Lowest HR',
    'HighSBP': 'Highest SBP',
    'LowSBP': 'Lowest SBP',
    'BIRTHDAY': 'Days old',
    'CNSInjury_No': 'No CNS Injury',
    'CNSInjury_Yes': 'Susp CNS Injury',
    'GCSWorstMotor_1': 'GCS Motor: 1',
    'GCSWorstMotor_2': 'GCS Motor: 2',
    'GCSWorstMotor_3': 'GCS Motor: 3',
    'GCSWorstMotor_4': 'GCS Motor: 4',
    'GCSWorstMotor_5': 'GCS Motor: 5',
    'GCSWorstMotor_6': 'GCS Motor: 6',
    'GCSWorstMotor_Unable to assess': 'GCS Motor: Unkn',
    'GCSWorstTotal_10': 'GCS: 10',
    'GCSWorstTotal_11': 'GCS: 11',
    'GCSWorstTotal_12': 'GCS: 12',
    'GCSWorstTotal_13': 'GCS: 13',
    'GCSWorstTotal_14': 'GCS: 14',
    'GCSWorstTotal_15': 'GCS: 15',
    'GCSWorstTotal_3': 'GCS: 3',
    'GCSWorstTotal_4': 'GCS: 4',
    'GCSWorstTotal_5': 'GCS: 5',
    'GCSWorstTotal_6': 'GCS: 6',
    'GCSWorstTotal_7': 'GCS: 7',
    'GCSWorstTotal_8': 'GCS: 8',
    'GCSWorstTotal_9': 'GCS: 9',
    'GCSWorstTotal_Unable to assess': 'GCS: Unkn',
    'GCSIntub_No': 'Not Intubated',
    'GCSIntub_Yes': 'Intubated',
    'LOCWorst_Coma (unresponsive)': 'LOC: Coma',
    'LOCWorst_No coma': 'LOC: Not Coma',
    'LOCWorst_Unable to assess': 'LOC: Unkn',
    'PupilWorst_Both non-reactive (> 3mm)': 'Pupils: Non-react',
    'PupilWorst_Both pupils < 3mm, cannot be scored': 'Pupils: pinpoint',
    'PupilWorst_Both reactive': 'Pupils: react',
    'PupilWorst_One non-reactive (> 3mm)': 'Pupils: 1x non-react',
    'PupilHypo_No': 'No Tx Hypotherm',
    'PupilHypo_Yes': 'Tx Hypotherm',
    'Sex_Female': 'Female',
    'Sex_Male': 'Male',
    'cpr': 'CPR performed',
    'cancer': 'Cancer dx',
    'trauma': 'Trauma dx',
    'FSSgood': 'Good FSS',
    'agecat_0 to <14 days': '0-14d',
    'agecat_1 month to <12 months': '1m-12m',
    'agecat_14 days to <1 month': '14d-1m',
    'agecat_>12 months': '>12m',
    'admitsourcecat_Direct: Referral Hosp': 'Direct Admit',
    'admitsourcecat_ED': 'ED Admit',
    'admitsourcecat_Inpatient Unit': 'Inpatient transfer',
    'admitsourcecat_OR/PACU': 'OR admit',
    'primarysyscat_CV/Resp': 'Primary Dx CV/Resp',
    'primarysyscat_Cancer': 'Primary Dx Cancer',
    'primarysyscat_Low Risk': 'Primary Dx Low Risk',
    'primarysyscat_Neurologic': 'Primary Dx Neuro',
    'primarysyscat_Other': 'Primary Dx Other'
    }
result = results["Mortality"]
rf = result["regressors"][0][0]
ylocal = result["yall"][0][0]
Xlocal = X.loc[ylocal.index.intersection(X.index)]
Xlocal.columns = [language_map[i] for i in Xlocal.columns]


explainer = shap.TreeExplainer(rf)
shap_values = explainer(Xlocal)


example1 = shap_values[0]
example1.base_values = example1.base_values[0]
example2 = shap_values[1]
example2.base_values = example2.base_values[0]

shap.plots.waterfall(example1)
shap.plots.waterfall(example2)

# Poor outcome
example3rf = shap_values[Xlocal.index.get_loc(1199)]
example3rf.base_values = example3rf.base_values[0]
shap.plots.waterfall(example3rf)
example4rf = shap_values[Xlocal.index.get_loc(7100)]
example4rf.base_values = example4rf.base_values[0]
shap.plots.waterfall(example4rf)

gb = result["regressors"][1][0]
explainer2 = shap.TreeExplainer(gb)
shap_values2 = explainer2(Xlocal)
example1 = shap_values2[0]
example1.base_values = example1.base_values[0]
example2 = shap_values2[1]
example2.base_values = example2.base_values[0]
example3 = shap_values2[2]
example3.base_values = example3.base_values[0]
example4 = shap_values2[3]
example4.base_values = example4.base_values[0]

shap.plots.waterfall(example1)
shap.plots.waterfall(example2)
shap.plots.waterfall(example3)
shap.plots.waterfall(example4)


# Poor outcome
example5 = shap_values2[Xlocal.index.get_loc(1199)]
example5.base_values = example5.base_values[0]
shap.plots.waterfall(example5)
example6 = shap_values2[Xlocal.index.get_loc(7100)]
example6.base_values = example6.base_values[0]
shap.plots.waterfall(example6)

# For paper
#
shap.plots.waterfall(example3,show=False)
fig = plt.gcf()
all_ax = fig.get_axes()
ax = all_ax[1]
fig.suptitle("(a) Waterfall Plot: Gradient Boosting prediction of Low Risk",fontsize=16)
ax.xaxis.set_label_position('bottom')
ax.set_xlabel('Model Prediction Contribution',fontsize=12)
plt.show()

shap.plots.waterfall(example6,show=False)
fig = plt.gcf()
all_ax = fig.get_axes()
ax = all_ax[1]
fig.suptitle("(b) Waterfall Plot: Gradient Boosting prediction of High Risk",fontsize=16)
ax.xaxis.set_label_position('bottom')
ax.set_xlabel('Model Prediction Contribution',fontsize=12)
plt.show()



ens = result["regressors"][2][0]
ylocal = result["yall"][2][0]
Xlocal = X.loc[ylocal.index.intersection(X.index)]
explainer3 = shap.KernelExplainer(ens.predict,Xlocal)
example5 = explainer3.shap_values(Xlocal.iloc[0,:])
shap.plots.force(explainer3.expected_value,example5,Xlocal.iloc[0,:])

explainer4 = shap.Explainer(ens.predict,Xlocal)
example6 = explainer4(Xlocal)

outdir = '../img/shap/'
shap.summary_plot(shap_values,show=False)
plt.title('SHAP Summary Plot for RF Model')
#plt.savefig(outdir+'rf_summary.png')
plt.show()
shap.summary_plot(shap_values2,show=False)
plt.title('SHAP Summary Plot for GB Model')
#plt.savefig(outdir+'gb_summary.png')
plt.show()
