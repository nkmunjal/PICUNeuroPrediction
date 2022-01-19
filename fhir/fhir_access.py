import asyncio
import os
from fhirpy import AsyncFHIRClient
from fhirpy import SyncFHIRClient
import datetime
import random
from .loinc_lookup import *
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

def datetime_to_str(timestamp):
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S-05:00")

def str_to_datetime(str_time):
    return datetime.datetime.strptime(str_time,"%Y-%m-%dT%H:%M:%S%z")


client = SyncFHIRClient('http://127.0.0.1:8080/fhir/',authorization='')
##
#resources = client.resources('Patient')
#resources = resources.search(name='Amy Allen').limit(10).sort('name')
#patients = resources.fetch()

##

#organization = client.resource('Organization',name='test.software2',active=False)
#organization.save()

##


def make_patient(client,first,last,gender):
    patient = client.resource("Patient",name={"given":first,"family":last},gender=gender)
    patient.save()
    return patient

##
#patients = client.resources("Patient").fetch()
#organizations = client.resources("Organization").fetch()
#results = client.resources("Observation").fetch()

##
#patient = patients[2]
#focused_results = client.resources("Observation").search(subject=patient['id']).fetch()

##
#sodium = {"coding": [ {"system":"http://loinc.org","code":"2951-2","display":"Serum Sodium mmol/L"}]}
#result1 = client.resource("Observation",subject={'reference':"Patient/"+patient['id']},code=sodium,valueQuantity={"value":142,"unit":"mmol/L"})
#result2 = client.resource("Observation",subject={'reference':patient['id']},code="2823-3",value="3.7")
#result1.save()
#result2.save()

##

def make_encounter(client, patient):
    timestamp = datetime.datetime.now()
    strf = datetime_to_str(timestamp)
    encounter = client.resource("Encounter",subject={'reference':"Patient/"+patient['id']},period={'start':strf})
    encounter.save()
    return encounter

#encounters = client.resources("Encounter").fetch()

##

def make_observation(client,encounter,obs_name,value):
    enc_date = str_to_datetime(encounter['period']['start'])
    obs_delta = datetime.timedelta(seconds=random.randint(1,60*60*2))
    obs_date = datetime_to_str(enc_date+obs_delta)
    patient = encounter['subject'].to_resource()
    loinc = name_to_loinc(obs_name)
    coding = {"system":"http://loinc.org","code":loinc,"display":obs_name}
    if type(value) == str:
        obs = client.resource("Observation",subject={'reference':"Patient/"+patient['id']}, code={"coding":coding},
                issued=obs_date, valueString=value,
                encounter={'reference':"Encounter/"+encounter['id']})
    else:
        unit_text = units[obs_name]
        obs = client.resource("Observation",subject={'reference':"Patient/"+patient['id']}, code={"coding":coding},
                issued=obs_date, valueQuantity={"value":value,"unit":unit_text},
                encounter={'reference':"Encounter/"+encounter['id']})
    obs.save()
    return obs

##

def make_full_patient_record(client):
    first_names = ["Amy","Bill","Charlie","Danielle","Eric","Faith","Gerald","Hattie","Iris","Jane","Katie","Leo","Mark","Nathan","Otto",
            "Priscilla","Quinn","Rose","Scott","Tina","Ulysses","Victoria","William","Youssef","Zoey"]
    last_names = ["Allen","Baker","Carter","Diaz","Edwards","Forsyth","Garcia","Hall","Ichikawa","Jagger","Kane","Logan","Machado","Nguyen","O'hara",
            "Packard","Quaid","Romanoff","Smith","Tucker","Uhlman","Vincente","Wall","Young","Zhou"]
    first = random.choice(first_names)
    last = random.choice(last_names)
    gender = random.choice(['male','female'])
    patient = make_patient(client,first,last,gender)
    encounter = make_encounter(client,patient)
    for key in model_vitals:
        for i in range(5):
            value = model_generator[key]()
            obs = make_observation(client,encounter,key,value)
    for key in model_exam:
        for i in range(5):
            value = model_generator[key]()
            obs = make_observation(client,encounter,key,value)
    for key in model_labs:
        for i in range(2):
            value = model_generator[key]()
            obs = make_observation(client,encounter,key,value)
    for key in model_other:
        value = model_generator[key]()
        obs = make_observation(client,encounter,key,value)
    return patient,encounter

def build_patient_from_encounter(encounter):
        patient = encounter['subject'].to_resource()
        first = " ".join(patient['name'][0]['given'])
        last = patient['name'][0]['family']
        return {'encounter':encounter,'resource':patient,'fname':first,'lname':last}

def get_encounters(client):
    encounters = client.resources("Encounter").fetch()
    patients = []
    for encounter in encounters:
        patients.append(build_patient_from_encounter(encounter))
    return patients

def choose_encounter(client):
    patients = get_encounters(client)
    i=0
    for p in patients:
        encounter = p['encounter']
        patient = p['resource']
        first = p['fname']
        last = p['lname']
        print("{:2.0f}: {} {}: {}".format(i,first,last,encounter['id']))
        i += 1 
    choice = -1
    while choice < 0 or choice >= len(patients):
        choice = int(input("Choice? "))
    return patients[choice]['encounter']

def get_values_for_encounter(encounter,variable):
    loinc_code = name_to_loinc(variable)
    obs = client.resources("Observation").search(encounter=encounter.to_reference()['reference'],code__contains=loinc_code).fetch()
    vals = []
    for resource in obs:
        if 'valueQuantity' in resource.keys():
            vals.append(resource['valueQuantity']['value'])
        elif 'valueString' in resource.keys():
            vals.append(resource['valueString'])
    return vals



def get_all_values_for_encounter(encounter,variable_list=None):
    if variable_list is None:
        variable_list = model_data_points
    big_list = {}
    for variable in variable_list:
        vals = get_values_for_encounter(encounter,variable)
        big_list[variable] = vals
    big_list['Sex'] = encounter['subject'].to_resource()['gender']
    return big_list

def get_low_high(data,var):
    print(var,data)
    if var in model_choices:
        # categorical
        if type(data) == str:
            return data,""
        data_num = [model_choices[var].index(i) for i in data]
        data_num.sort()
        return model_choices[var][data_num[0]],""
    else:
        data.sort()
        return "{:.2f}".format(data[0]),"{:.2f}".format(data[-1])

##
def generate_patient_row(patient_data):
    row = []
    for column in dataframe_all_columns:
        if column == 'CNSInjury_No':
            row.append(0)
        elif column == 'CNSInjury_Yes':
            row.append(1)
        elif column in dataframe_continuous_columns:
            # continuous measure
            decoder = dataframe_continuous_columns[column]
            if decoder[0] not in patient_data:
                row.append(np.nan)
            else:
                patient_data[decoder[0]].sort(reverse = decoder[1])
                row.append(patient_data[decoder[0]][0])
        elif column in dataframe_categorical_columns:
            # categorical measure
            decoder = dataframe_categorical_columns[column]
            if decoder[0] not in patient_data:
                row.append(np.nan)
            else:
                if type(patient_data[decoder[0]]) == list:
                    if type(patient_data[decoder[0]][0]) == str:
                        # str
                        ordered_data = [model_choices[decoder[0]].index(i) for i in patient_data[decoder[0]]]
                        ordered_data.sort(reverse = decoder[1])
                        row.append(1 if model_choices[decoder[0]][ordered_data[0]] == decoder[2] else 0)
                    else:
                        patient_data[decoder[0]].sort(reverse = decoder[1])
                        row.append(1 if patient_data[decoder[0]][0] == decoder[2] else 0)
                else:
                    #str
                    row.append(1 if patient_data[decoder[0]] == decoder[2] else 0)
        elif column in dataframe_pupil_columns:
            decoder = dataframe_pupil_columns[column]
            if 'Left'+decoder[0] not in patient_data or 'Right'+decoder[0] not in patient_data:
                row.append(np.nan)
            else:
                left_ordered = [model_choices['Left'+decoder[0]].index(i) for i in patient_data['Left'+decoder[0]]]
                right_ordered = [model_choices['Right'+decoder[0]].index(i) for i in patient_data['Right'+decoder[0]]]
                left_ordered.sort(reverse = decoder[1])
                right_ordered.sort(reverse = decoder[1])
                sum = 0
                if model_choices['Left'+decoder[0]][left_ordered[0]] == decoder[3]:
                    sum += 1
                if model_choices['Right'+decoder[0]][right_ordered[0]] == decoder[3]:
                    sum += 1
                row.append(1 if sum == decoder[2] else 0)
        else:
            print("THIS SHOULD NOT BE REACHED")
            print(column)
            raise Exception
    return pd.DataFrame([row],columns=dataframe_all_columns)
##

def preprocess_row(row):
    pt = os.path.dirname(os.path.realpath(__file__))
    model_statsdf = pd.read_csv(os.path.join(pt,'column_statistics.csv'),index_col=0)
    newrow = row.fillna(model_statsdf.loc['median'])
    cont_cols = list(dataframe_continuous_columns.keys())
    newrow[cont_cols] = (newrow[cont_cols] - model_statsdf[cont_cols].loc['mean'])/model_statsdf[cont_cols].loc['std']
    return newrow

def get_plot(processed_row):
    pt = os.path.dirname(os.path.realpath(__file__))
    mort_model = joblib.load(os.path.join(pt,"./random_forest_mortality.joblib"))
    morbmort_model = joblib.load(os.path.join(pt,"./random_forest_morbmort.joblib"))
    mortprediction = mort_model.predict(processed_row)
    morbmortprediction = morbmort_model.predict(processed_row)


    fig = plt.figure(figsize=(8,4),dpi=96)
    ax0 = fig.add_subplot(211)

    mortexplainer = shap.TreeExplainer(mort_model)
    mort_shap_values = mortexplainer(processed_row)
    mort_expl = mort_shap_values[0]
    mort_expl.base_values = mort_expl.base_values[0]
    shap.plots.waterfall(mort_expl,show=False)
    ax0.set_title("Mortality prediction")

    morbmortexplainer = shap.TreeExplainer(morbmort_model)
    morbmort_shap_values = morbmortexplainer(processed_row)
    ax1 = fig.add_subplot(212)
    morbmort_expl = morbmort_shap_values[0]
    morbmort_expl.base_values = morbmort_expl.base_values[0]
    shap.plots.waterfall(morbmort_expl,show=False)
    ax1.set_title("Morbidity/Mortality prediction")
    fig.set_tight_layout({'w_pad':2})
    return [fig,[mortprediction,morbmortprediction]]

def run_prediction(processed_row):
    fig,preds = get_plot(processed_row)
    mortprediction,morbmortprediction = preds
    print("Predicted output for mortality: {}".format(mortprediction))
    print("Predicted output for morb/mort: {}".format(morbmortprediction))
    plt.show()

def make_custom_patient():
    categories = list(model_generator.keys())
    patient_data = {}
    for cat in categories:
        if cat in model_choices:
            i = 0
            selection = -1
            while selection < 0 or selection >= len(model_choices[cat]):
                print("{}:".format(cat))
                for choice in model_choices[cat]:
                    print("  {:2.0f}: {}".format(i,choice))
                    i+=1
                selection = input("Choice (Blank if unavailable)? ")
                if selection == '':
                    break
                else:
                    selection = int(selection)
            if selection != '':
                patient_data[cat] = [model_choices[cat][selection]]
        else:
            val = input("{}: value {} (Blank if unavailable)? ".format(cat,units[cat]))
            if val != '':
                patient_data[cat] = [float(val)]
    return generate_patient_row(patient_data)

def run_it_all(client):
    enc = choose_encounter(client)
    patient_data = get_all_values_for_encounter(enc)
    row = generate_patient_row(patient_data)
    processed_row = preprocess_row(row)
    run_prediction(processed_row)

##

if __name__ == '__main__':
    choice = int(input("(1) FHIR or (2) custom? "))
    if choice == 1:
        run_it_all(client)
    elif choice == 2:
        row = make_custom_patient()
        processed_row = preprocess_row(row)
        run_prediction(processed_row)

