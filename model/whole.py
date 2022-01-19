#!/usr/bin/python

# @author Neil Munjal
# @date 2019-07-27


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score,matthews_corrcoef
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.utils import resample
from sklearn.base import clone
import statsmodels.api as sm
import statsmodels.formula.api as smf
from imblearn.over_sampling import SMOTE
#import tensorflow as tf

def bootstrap_regressor(features,labels,regressor=None, train_size=1, runs=50,oversample=False):
    if regressor == None:
        linear = True
    else:
        linear = False
    aurocs = []
    auprcs = []
    regressors = []
    ys = []
    yhats = []
    for i in range(0,runs):
        boot = resample(features.index,replace=True,random_state=i,n_samples=int((train_size)*len(features)))
        oob = pd.Series([i for i in features.index if i not in boot])
        X = features.loc[boot]
        y = labels.loc[boot]
        if oversample:
            k = 2
            smo = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=random_state)
            X,y = smo.fit_resample(X,y)
        Xtest = features.loc[oob]
        ytest = labels.loc[oob]
        if linear:
            data = X.join(y.astype(int))
            regressor = smf.glm(formula = "{} ~ agecat + admitsourcecat + cpr + cancer + trauma + primarysyscat + FSSgood + Q('neurologic.score') + Q('nonneurologic.score')".format(y.name),
                    family=sm.families.Binomial(),
                    data=data).fit()
        else:
            regressor = clone(regressor)
            regressor.fit(X,y)
        yhat = pd.Series(regressor.predict(Xtest),index=oob)
        auc = roc_auc_score(ytest,yhat)
        avg_prec = average_precision_score(ytest,yhat)
        aurocs.append(auc)
        auprcs.append(avg_prec)
        regressors.append(regressor)
        yhats.append(yhat)
        ys.append(ytest)
    return aurocs,auprcs,regressors,ys,yhats

def calibration_curve(y,ypred,label="",style=0,plot=False):
    deciles = np.linspace(0,1,11)
    hlplot = []
    hl_stat = 0
    for ind,decile in enumerate(deciles):
        if ind >= len(deciles)-1:
            break
        next_decile = deciles[ind+1]
        if ind == len(deciles)-2:
            next_decile=1.0
        filter = (ypred > decile) & (ypred <= next_decile)
        if ind==0:
            filter = ypred <= next_decile
        perc = (decile+next_decile)/2
        total = len(y[filter])
        obs_true = y[filter].sum()
        obs_not_true = total-obs_true
        exp_true = perc*total
        exp_not_true = total-exp_true
        #print("{:.1} - {:.1}: {} {}".format(decile,next_decile,y[filter].sum(),len(y[filter])))
        if plot:
            print("{:.1} - {:.1}: total: {} true: {} false: {} exp_true: {}, exp_false: {}".format(decile,next_decile,total,obs_true,obs_not_true,exp_true,exp_not_true))
        if exp_true != 0 and exp_not_true != 0:
            hl_stat += (obs_true-exp_true)**2/exp_true + (obs_not_true-exp_not_true)**2/exp_not_true
        hlplot.append(y[filter].sum()/len(y[filter]))
    if plot:
        linestyle = ['solid','dashed','dotted','solid']
        color = ['blue','orange','green','black']
        plt.plot(deciles[:-1]+0.05,hlplot,linestyle=linestyle[style],color=color[style],alpha=0.9,label=label)
        plt.plot([0,1],[0,1])
        print("HL stat: {}. p: {}".format(hl_stat,1-stats.chi2.cdf(hl_stat,len(deciles)-2)))
    return hlplot,hl_stat

##

def calibration_confidence_interval(yall,yhatall,labels,styles,title="",prefix=""):
    if prefix != "":
        prefix = prefix +" "
    deciles = np.linspace(0,1,11)
    linestyle = ['dotted','dashed','solid','solid']
    color = ['red','blue','darkgoldenrod','black']
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    ax.plot([0,1],[0,1],color='black')
    plt.xticks(deciles[:-1]+0.05,["{:.1f}-{:.1f}".format(i,i+.1) for i in deciles[:-1]])
    for ys,yhats, label, style in zip(yall,yhatall,labels,styles):
        scoresarr = []
        hlstats = []
        for y,yhat in zip(ys,yhats):
            score,stat = calibration_curve(y,yhat,"GBM")
            scoresarr.append(score)
            hlstats.append(stat)
        scores = np.array(scoresarr)
        hl_stat = np.median(hlstats)
        print("HL stat: {}. p: {}".format(hl_stat,1-stats.chi2.cdf(hl_stat,len(deciles)-2)))
        lower = []
        mid = []
        upper = []
        for i in range(0,scores.shape[1]):
            scores[:,i].sort()
            lower.append( scores[:,i][5]) # 1 = 95% confidence interval, 5 = ~85% CI
            upper.append(scores[:,i][-6])
            mid.append(scores[:,i][scores.shape[0]//2])
        ax.plot(deciles[:-1]+0.05,mid,linewidth=2,linestyle=linestyle[style],color=color[style],alpha=0.9,label=label)
        ax.fill_between(deciles[:-1]+0.05, lower, upper, color=color[style], alpha=.05)
    leg = ax.legend(loc='best', shadow=True, fontsize='medium')
    plt.title("{}Calibration Curves - {}".format(prefix,title))
    plt.xlabel("Estimated (decile)")
    plt.ylabel("Actual (decile)")
    return fig,ax

def plot_roc_curves(yall,yhatall,labels,styles,aurocall,title="",prefix=""):
    if prefix != "":
        prefix = prefix + " "
    linestyle = ['dotted','dashed','solid','solid']
    color = ['red','blue','darkgoldenrod','black']
    base_fpr = np.linspace(0, 1, 101)
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    ax.plot([0,1],[0,1],color='black')
    for ys,yhats,label,style,aurocs in zip(yall,yhatall,labels,styles,aurocall):
        tprs = []
        for y,yhat in zip(ys,yhats):
            fpr,tpr, _ = roc_curve(y, yhat)
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0
            tprs.append(tpr)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        lower_tprs = np.percentile(tprs,7.5,axis=0)
        upper_tprs = np.percentile(tprs,92.5,axis=0)
        auroc = np.mean(aurocs)
        lab="{} ({:.2f})".format(label,auroc)
        plt.plot(base_fpr,mean_tprs,color=color[style],linewidth=2,linestyle=linestyle[style],label=lab)
        ax.fill_between(base_fpr, lower_tprs, upper_tprs, color=color[style], alpha=.05)
    plt.title("{}Receiver Operating Characteristic - {}".format(prefix,title))
    plt.xlabel("1-Specificity (Power)")
    plt.ylabel("Sensitivity")
    ax.legend(loc='best', shadow=True, fontsize='medium')
    return fig,ax

def plot_precision_recall(yall,yhatall,labels,styles,auprcall=-1,title="",prefix=""):
    if prefix != "":
        prefix = prefix +" "
    linestyle = ['dotted','dashed','solid','solid']
    color = ['red','blue','darkgoldenrod','black']
    #base_recall = np.linspace(0, 1, 101)
    base_recall = np.linspace(0.0, 1.0, num=10)
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    for ys,yhats,label,style,auprcs in zip(yall,yhatall,labels,styles,auprcall):
        precs = []
        recs = []
        for y,yhat in zip(ys,yhats):
            prec, recall, _ = precision_recall_curve(y,yhat)
            # take a running maximum over the reversed vector of precision values, reverse the
            # result to match the order of the recall vector
            decreasing_max_precision = np.maximum.accumulate(prec)
            prec_interp = np.interp(base_recall,recall[::-1],decreasing_max_precision[::-1])[::-1]
            precs.append(prec_interp)
            #precs.append(decreasing_max_precision)
            #precs.append(prec)
            recs.append(recall)
        precs = np.array(precs)
        mean_precs = precs.mean(axis=0)
        lower_precs = np.percentile(precs,7.5,axis=0)
        upper_precs = np.percentile(precs,92.5,axis=0)
        #plt.step(base_recall,mean_precs,color=color[style],linewidth=2,linestyle=linestyle[style],alpha=0.9, where='post',label=label)
        auprc = np.mean(auprcs)
        lab="{} ({:.2f})".format(label,auprc)
        plt.step(base_recall[::-1],mean_precs,color=color[style],linewidth=2,linestyle=linestyle[style],alpha=0.9, where='post',label=lab)
        ax.fill_between(base_recall[::-1], lower_precs, upper_precs, color=color[style], alpha=.05)
    plt.title("{}Precision Recall Curves - {}".format(prefix,title))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    ax.legend(loc='best', shadow=True, fontsize='medium')
    return fig,ax

##

if __name__ == '__main__':
    verbose = False
    data_dir = '/home/repositories/git/nkmj/topicc-local/data/'
    files = ['CARDIACPROCEDURES.csv', 'CARDIACSURGERY.csv', 'CAREPROCESSES.csv', 'CATASTROPHICEVENTS.csv',
        'CPR.csv', 'DEATH.csv', 'HOSPITALADMIT_CE.csv', 'HOSPITALADMIT.csv', 'HOSPITALDISCHARGE.csv',
        'LIMITOFCARE.csv', 'PHYSIOSTATUS.csv', 'PICUADMIT_ADMITSECONDDX.csv', 'PICUADMIT_AHD.csv',
        'PICUADMIT_CHD.csv', 'PICUADMIT_CHRONICDX.csv', 'PICUADMIT.csv', 'PICUDISCHARGE.csv',
        'PICUDISCHARGE_DISCHRONICDX.csv', 'PICUDISCHARGE_DISSECONDDX.csv', 'SURGERY.csv']

    adm = data_dir+'HOSPITALADMIT.csv'
    adm_cols = ['PudID','BIRTHDAY','Sex']
    adm_FSS_cols = ['BaselineFSSCommun', 'BaselineFSSFeeding', 'BaselineFSSMental', 'BaselineFSSMotor',
        'BaselineFSSResp', 'BaselineFSSSensory']
    adm_cols.extend(adm_FSS_cols)
    prism = data_dir+'PHYSIOSTATUS.csv'
    dc = data_dir+'HOSPITALDISCHARGE.csv'
    dc_cols = ['PudID','HospDisAlive']
    dc_FSS_cols = ['HospDisFSSCommun', 'HospDisFSSFeeding', 'HospDisFSSMental', 'HospDisFSSMotor',
        'HospDisFSSResp', 'HospDisFSSSensory']
    dc_cols.extend(dc_FSS_cols)

    picudc = data_dir+'PICUDISCHARGE.csv'
    picudc_cols = ['PudID','PICUDisPrimaryDx']

    df = pd.read_csv(picudc)
    picudcdf = df[picudc_cols].copy()
    picudcdf.set_index(picudc_cols[0],inplace=True)
    neuro_items = ['Seizures','Neurological CSF related (hydrocephalus / Chiari / fenestrations / arachnoid cysts)',
            'Neurological miscellaneous','Neurological - cords, bones','Neurological - vascular malformations',
            'Stroke / Cerebral Ischemia / Cerebral infarction','Central nervous system infection']
    # Naive filtering of discharge diagnosis; unfortunately many patients who died
    # did not get a PICU dc diagnosis filled out
    discharge_dx_filter = picudcdf['PICUDisPrimaryDx'].isin(neuro_items)

    fss_text_to_num = {"Normal":1,"Mild dysfunction":2,"Moderate dysfunction":3,"Severe dysfunction":4,"Very severe dysfunction":5,np.nan:100}

    df = pd.read_csv(adm)

    admdf = df[adm_cols].copy()
    admdf.set_index(adm_cols[0],inplace=True)
    admfssdf = admdf[adm_FSS_cols].apply(lambda x: x.apply(lambda y: fss_text_to_num[y]))
    admfssdf['BaselineFSSSum'] = admfssdf.sum(axis=1)

    df = pd.read_csv(dc)
    dcdf = df[dc_cols].copy()
    dcdf.set_index(dc_cols[0],inplace=True)
    dcdf_num = dcdf[dc_FSS_cols].apply(lambda x: x.apply(lambda y: fss_text_to_num[y]))
    dcdf_num['HospDisFSSSum'] = dcdf_num.sum(axis=1)
    dcdf_num['HospDisAlive'] = dcdf['HospDisAlive']

    bigdf = pd.merge(admfssdf,dcdf_num,on='PudID')
    bigdf['FSSDiff'] = bigdf['HospDisFSSSum']-bigdf['BaselineFSSSum']
    #plt.hist(bigdf['FSSDiff'],bins=range(-5,30,1))
    #plt.show()

    ###
    # PRISM
    ###

    df = pd.read_csv(prism)
    df = df.set_index('PudID')

    #prism_cols=['siteid', 'LowpH', 'HighpH', 'HighPCO2', 'LowPa02',
    prism_cols=['LowpH', 'HighpH', 'HighPCO2', 'LowPa02',
           'LowIonCalcium', 'HighIonCalcium', 'LowSodium',
           'HighSodium', 'HighPotassium', 'HighBUN',
           'HighCreatinine', 'LowGlucose',
           'HighGlucose', 'LowCO2', 'HighCO2',
           'LowTotalCalcium', 'HighTotalCalcium', 'LowWBC',
           'HighWBC', 'LowHemoglobin', 'HighHemoglobin',
           'LowPlatelets', 'HighPT', 'HighPTT',
           'HighINR', 'HighTemp', 'LowTemp', 'HighRespRate',
           'LowRespRate', 'HighHeartRate', 'LowHeartRate', 'HighSBP', 'LowSBP',
           'CNSInjury', 'GCSWorstMotor', 'GCSWorstTotal', 'GCSIntub', 'LOCWorst',
           'PupilWorst', 'PupilHypo']

    prism_cols_cont=[ 'LowpH', 'HighpH', 'HighPCO2', 'LowPa02',
           'LowIonCalcium', 'HighIonCalcium', 'LowSodium',
           'HighSodium', 'HighPotassium', 'HighBUN',
           'HighCreatinine', 'LowGlucose',
           'HighGlucose', 'LowCO2', 'HighCO2',
           'LowTotalCalcium', 'HighTotalCalcium', 'LowWBC',
           'HighWBC', 'LowHemoglobin', 'HighHemoglobin',
           'LowPlatelets', 'HighPT', 'HighPTT',
           'HighINR', 'HighTemp', 'LowTemp', 'HighRespRate',
           'LowRespRate', 'HighHeartRate', 'LowHeartRate', 'HighSBP', 'LowSBP','BIRTHDAY']

    #onehot: siteid, CNSInjury, GCSIntub, LOCWorst, PupilWorst, PupilHypo
    #for col in prism_cols:
        #print(df[col].head(5))
        #print(df[col].value_counts())
        #input('Next?')

    model_cols = prism_cols + ['BIRTHDAY','Sex']
    df['BIRTHDAY']=-admdf['BIRTHDAY']
    df['Sex']=admdf['Sex']
    prismdf = pd.get_dummies(df[model_cols]) # One hot encoding
    mapperdf = pd.DataFrame([prismdf[prism_cols_cont].mean(),prismdf[prism_cols_cont].std(),prismdf.median()],index=['mean','std','median'])
    prismdf[prism_cols_cont] = (prismdf[prism_cols_cont]-prismdf[prism_cols_cont].mean())/prismdf[prism_cols_cont].std() # normalization
    if verbose:
        print(prismdf)
    imputer = SimpleImputer(strategy='median') # median imputation
    #imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=5)) # kNN imputation
    prismdf_imputed = pd.DataFrame(imputer.fit_transform(prismdf),columns=prismdf.columns,index=prismdf.index)
    if verbose:
        print(prismdf_imputed)
    prismdf=prismdf_imputed

    alivelabel = dcdf['HospDisAlive'] == 'Yes'
    newmorbidlabel = bigdf['FSSDiff'] >= 3

    outcomeframe = { 'nonew': ((~newmorbidlabel) & alivelabel),
            'new': (newmorbidlabel & alivelabel),
            'died': ~alivelabel
            }

    outcomedf = pd.DataFrame(outcomeframe)
    outcome_to_num = {'nonew':0,'new':1,'died':2}
    singleoutcomedf = outcomedf.idxmax(axis=1).apply(lambda x: outcome_to_num[x])



    #Analysis begins:
    #prismdf has one-hot encoded PRISM data with imputed filling of nan
    #outcomedf has one-hot encoded outcomes series: nonew, new, died boolean; other options include multivariate outcomes in singleoutcomedf
    myfilter = prismdf['CNSInjury_Yes']==True # Neurological injury patients only
    #myfilter = prismdf.index != -1

    Xprism = prismdf.loc[myfilter]
    X = Xprism
    ymort = outcomedf['died'][myfilter]
    ycomb = ~outcomedf['nonew'][myfilter]
    y=ymort
    random_state = 41

    #auc,avg_prec,rf = run_rfc(X,y,random_state=random_state) # outcomedf['died'], outcomedf['new'], singleoutcomedf, singleoutcomedf > 0
    #print(auc,avg_prec)
    #joblib.dump(rf,"./random_forest.joblib")


    #demonstrate that training on the whole dataset gives an overfitted model
    #rf = RandomForestRegressor(n_estimators = 150, random_state = 42)
    #rf.fit(X,y)
    #pred_rf = pd.Series(rf.predict(X),index=y.index)
    #auc = roc_auc_score(y,pred_rf)
    #avg_prec = average_precision_score(y,pred_rf)
##
    ptdf = pd.read_csv('/home/repositories/git/nkmj/topicc-local/tables/prism_topicc.csv',index_col="PudID")
    ptdf_orig = ptdf.copy(deep=True)
    catcols = ['agecat','admitsourcecat','primarysyscat']
    for col in catcols:
        ptdf = ptdf.join(pd.get_dummies(ptdf[col],prefix=col))
        ptdf.drop(col,axis=1,inplace=True)
    ptdf.drop(["mortality","morbidity"],axis=1,inplace=True)
    Xpt = ptdf_orig.loc[myfilter]
    model = smf.glm(formula = "{} ~ agecat + admitsourcecat + cpr + cancer + trauma + primarysyscat + FSSgood + Q('neurologic.score') + Q('nonneurologic.score')".format(y.name),
            family=sm.families.Binomial(),
            data=Xpt.join(y.astype(int))).fit()
    pred = model.predict(Xpt)
    auc = roc_auc_score(y,pred)
    prc = average_precision_score(y,pred)
    print("{} {}".format(auc,prc))
    Xpt_cat = ptdf.loc[myfilter]
    Xbig = Xprism.join(Xpt_cat.drop(["neurologic.score","nonneurologic.score"],axis=1))

    X = Xbig
    rf_top_features=['GCSWorstTotal_3','LowTemp','PupilWorst_Both non-reactive (> 3mm)',
            'HighPTT','HighINR','LowSBP','HighSBP','LowHeartRate','HighPT',
            'LowSodium','HighSodium','LowRespRate','HighBUN','HighPotassium',
            'LowCO2','HighHeartRate','HighHemoglobin','LowIonCalcium',
            'HighpH','HighCreatinine']
    gb_top_features=['GCSWorstTotal_3','GCSWorstMotor_1','HighINR','LOCWorst_Coma (unresponsive)',
            'LowSBP','PupilWorst_Both non-reactive (> 3mm)','LowTemp','HighPT',
            'HighPTT','HighHeartRate','HighCreatinine','HighSodium','PupilWorst_Both reactive',
            'LowSodium','HighpH','HighSBP','cancer','HighCO2','LowIonCalcium','HighTemp']
    #X = Xbig[gb_top_features[:3]] # Model parsimony test; top_features chosen from RF/GB beeswarm SHAP plots
    runs = [("Mortality","mort",ymort),("Morbidity or Mortality","morbmort",ycomb)]
    results = {}
    regressor_list = ["RF","GB","Ensemble","LR"]
    #regressor_list = ["GB"]

    for category,prefix,y in runs:
        print("Starting {}".format(category))
        inner_results = {}
        rf = RandomForestRegressor(n_estimators = 150,max_depth=6, random_state = random_state)
        gb = GradientBoostingRegressor(n_estimators=30, learning_rate=0.15, max_features='sqrt',max_depth=12)
        models = [
                ('rf',RandomForestRegressor(n_estimators = 150, max_depth=6, random_state = random_state)),
                ('xgb',GradientBoostingRegressor(n_estimators=30, learning_rate=0.15, max_features='sqrt',max_depth=12)),
                ('svm',SVR())
                ]
        ensemble = VotingRegressor(models)
        rocs = []
        prcs = []
        regressors = []
        yall = []
        yhatall = []
        aurocs, auprcs,rfs,ys,yhats = bootstrap_regressor(X,y,rf)
        #aurocs, auprcs,rfs,ys,yhats = bootstrap_regressor(X,y,rf,oversample=True)
        rocs.append(aurocs)
        prcs.append(auprcs)
        regressors.append(rfs)
        yall.append(ys)
        yhatall.append(yhats)
        aurocs, auprcs,gbs,ys,yhats = bootstrap_regressor(X,y,gb)
        rocs.append(aurocs)
        prcs.append(auprcs)
        regressors.append(gbs)
        yall.append(ys)
        yhatall.append(yhats)
        aurocs, auprcs,ens,ys,yhats = bootstrap_regressor(X,y,ensemble)
        rocs.append(aurocs)
        prcs.append(auprcs)
        regressors.append(ens)
        yall.append(ys)
        yhatall.append(yhats)
        aurocs, auprcs,glms,ys,yhats = bootstrap_regressor(Xpt,y)
        rocs.append(aurocs)
        prcs.append(auprcs)
        regressors.append(glms)
        yall.append(ys)
        yhatall.append(yhats)
        inner_results["rocs"]=rocs
        inner_results["prcs"]=prcs
        inner_results["regressors"]=regressors
        inner_results["yall"]=yall
        inner_results["yhatall"]=yhatall
        inner_results["prefix"]=prefix
        results[category]=inner_results
        #aurocs, auprcs,svs,ys,yhats = bootstrap_regressor(X,y,sv)
        print("Finished {}".format(category))
##

    for key,result in results.items():
        for i in range(len(regressor_list)):
            for stat in ['rocs','prcs']:
                model_name = regressor_list[i]
                ans = result[stat][i]
                ans.sort()
                print("{}: {:.2f} ({:.2f} - {:.2f})".format(model_name,ans[25],ans[2],ans[-3]))


##
    subfigure = {'Mortality':'(a)','Morbidity or Mortality':'(b)'}
    for category,inner_results in results.items():
        rocs = inner_results["rocs"]
        prcs = inner_results["prcs"]
        regressors = inner_results["regressors"]
        yall = inner_results["yall"]
        yhatall = inner_results["yhatall"]
        prefix = inner_results["prefix"]
        fig,ax = plt.subplots(2,1)
        fig.set_size_inches(8,6)
        ax[0].boxplot(rocs)
        ax[0].set_title("{} AUROC - {}".format(subfigure[category],category))
        ax[0].set_xticks([1,2,3,4])
        ax[0].set_xticklabels(["RF","GB","Ens","LR"])
        ax[1].boxplot(prcs)
        ax[1].set_title("{} AUPRC - {}".format(subfigure[category],category))
        ax[1].set_xticks([1,2,3,4])
        ax[1].set_xticklabels(["RF","GB","Ens","LR"])
        fig.savefig("/tmp/{}_aucs.png".format(prefix),dpi=300)
        plt.show()

        fig,ax = calibration_confidence_interval(yall,yhatall,regressor_list,[0,1,2,3],category,subfigure[category])
        fig.savefig("/tmp/{}_calibration.png".format(prefix),dpi=300)
        fig,ax = plot_roc_curves(yall,yhatall,regressor_list,[0,1,2,3],rocs,category,subfigure[category])
        fig.savefig("/tmp/{}_roc_curves.png".format(prefix),dpi=300)
        fig,ax = plot_precision_recall(yall,yhatall,regressor_list,[0,1,2,3],prcs,category,subfigure[category])
        fig.savefig("/tmp/{}_prc_curves.png".format(prefix),dpi=300)
