library(plyr)
library(dplyr)
library(pROC)
library(PRROC)
#Load packages

#Import files
setwd("/home/repositories/git/nkmj/topicc-local/data")

hospadmit <- read.csv("HOSPITALADMIT.csv")
picudischarge <- read.csv("PICUDISCHARGE.csv")
picuadmit <- read.csv("PICUADMIT.csv")
labsvitals <- read.csv("PHYSIOSTATUS.csv")
hospdc <- read.csv("HOSPITALDISCHARGE.csv")
cpr <- read.csv("CPR.csv")
ce <- read.csv("HOSPITALADMIT_CE.csv")
hospce <- read.csv("CATASTROPHICEVENTS.csv")
picuseconddx <- read.csv("PICUADMIT_ADMITSECONDDX.csv")
picuchrondx <- read.csv("PICUADMIT_CHRONICDX.csv")

#Assign numeric values to FSS
FSSbase <- select(hospadmit,c("PudID","BaselineFSSMental","BaselineFSSSensory","BaselineFSSCommun",
                              "BaselineFSSMotor","BaselineFSSFeeding","BaselineFSSResp"))

FSSbase_num <- lapply(FSSbase, function(i)
      ifelse(i == "Normal",1,
         ifelse(i == "Mild dysfunction",2,
                ifelse(i == "Moderate dysfunction",3,
                       ifelse(i == "Severe dysfunction",4,
                              ifelse(i == "Very severe dysfunction",5,i)))))
)
FSSbase_num <- as.data.frame(lapply(FSSbase_num,function(i) as.numeric(i)))
FSSbase_num$FSSbasesum <- rowSums(FSSbase_num[,c("BaselineFSSMental","BaselineFSSSensory","BaselineFSSCommun",
                                                 "BaselineFSSMotor","BaselineFSSFeeding","BaselineFSSResp")])

FSSdc <- select(hospdc,c("PudID","HospDisFSSMental","HospDisFSSSensory","HospDisFSSCommun",
                              "HospDisFSSMotor","HospDisFSSFeeding","HospDisFSSResp"))

#Build PRISM table
prism <- select(hospadmit,c(PudID, c(BIRTHDAY)))

prism$age <- with(prism,ifelse(-14<BIRTHDAY & BIRTHDAY<=0,"0 to <14 days",
                               ifelse(-30.4<BIRTHDAY & BIRTHDAY<=-14,"14 days to <1 month",
                                      ifelse(-365.25<BIRTHDAY & BIRTHDAY<=-30.4,"1 month to <12 months",
                                             ifelse(BIRTHDAY<=-365.25 & BIRTHDAY>=-4377.6,">12 months to 144 months",
                                                    ifelse(BIRTHDAY<-4377.6,">144 months",""))))))

prism <- left_join(prism, labsvitals, by="PudID")

#SBP
prism <- prism %>% mutate(.,nonneuro.sbp = with(., case_when(
  ((age == "0 to <14 days" | age == "14 days to <1 month") & LowSBP>55)~0,
  ((age == "0 to <14 days" | age == "14 days to <1 month") & LowSBP>=40 & LowSBP<=55)~3,
  ((age == "0 to <14 days" | age == "14 days to <1 month") & LowSBP<40)~7,
  (age == "1 month to <12 months" & LowSBP>65)~0,
  (age == "1 month to <12 months" & LowSBP<=65 & LowSBP>=45)~3,
  (age == "1 month to <12 months" & LowSBP<45)~7,
  (age == ">12 months to 144 months" & LowSBP>75)~0,
  (age == ">12 months to 144 months" & LowSBP<=75 & LowSBP>=5)~3,
  (age == ">12 months to 144 months" & LowSBP<55)~7,
  (age == ">144 months" & LowSBP>85)~0,
  (age == ">144 months" & LowSBP<=85 & LowSBP>=65)~3,
  (age == ">144 months" & LowSBP<65)~7
)))
prism$nonneuro.sbp <- with(prism,ifelse(is.na(nonneuro.sbp),0,nonneuro.sbp))

#HR
prism <- prism %>% mutate(.,nonneuro.hr = with(., case_when(
  ((age == "0 to <14 days" | age == "14 days to <1 month") & HighHeartRate<215)~0,
  ((age == "0 to <14 days" | age == "14 days to <1 month") & HighHeartRate>=215 & HighHeartRate<=225)~3,
  ((age == "0 to <14 days" | age == "14 days to <1 month") & HighHeartRate>225)~4,
  (age == "1 month to <12 months" & HighHeartRate<215)~0,
  (age == "1 month to <12 months" & HighHeartRate>=215 & HighHeartRate<=225)~3,
  (age == "1 month to <12 months" & HighHeartRate>225)~4,
  (age == ">12 months to 144 months" & HighHeartRate<185)~0,
  (age == ">12 months to 144 months" & HighHeartRate>=185 & HighHeartRate<=205)~3,
  (age == ">12 months to 144 months" & HighHeartRate>205)~4,
  (age == ">144 months" & HighHeartRate<145)~0,
  (age == ">144 months" & HighHeartRate<=155 & HighHeartRate>=145)~3,
  (age == ">144 months" & HighHeartRate>155)~4
)))
prism$nonneuro.hr <- with(prism,ifelse(is.na(nonneuro.hr),0,nonneuro.hr))

#Temp
prism$nonneuro.temp <- with(prism,ifelse(LowTemp<33 | HighTemp>40,3,0))
prism$nonneuro.temp <- with(prism,ifelse(is.na(LowTemp) & is.na(HighTemp),0,nonneuro.temp))

#Acidosis
prism <- prism %>% mutate(.,nonneuro.acidosis = with(.,case_when(
  (pHND == "Not Done" & CO2ND == "Not Done")~0,
  ((LowpH>7.28 & LowCO2>16.9) | (LowpH>7.28 & CO2ND == "Not Done") | (pHND == "Not Done" & LowCO2>16.9))~0,
  ((LowpH>=7.0 & LowpH<=7.28) | (LowCO2>=5 & LowCO2<=16.9))~2,
  (LowpH<7 | LowCO2<5)~6
)))

#Total CO2
prism$nonneuro.totalco2 <- with(prism,ifelse(HighCO2<=34 | CO2ND == "Not Done",0,4))

#pH Alkalosis
prism <- prism %>% mutate(.,nonneuro.ph = with(.,case_when(
  (pHND == "Not Done" | HighpH<7.48)~0,
  (HighpH>=7.48 & HighpH <=7.55)~2,
  (HighpH>7.55)~3
)))

#PaO2
prism <- prism %>% mutate(.,nonneuro.pao2 = with(.,case_when(
  (PaO2ND == "Not Done" | LowPa02>49.9)~0,
  (LowPa02<=49.9 & LowPa02>=42.0)~3,
  (LowPa02<42)~6
)))

#PCO2
prism <- prism %>% mutate(.,nonneuro.pco2 = with(.,case_when(
  (PCO2ND == "Not Done" | HighPCO2<50)~0,
  (HighPCO2>=50 & HighPCO2<=75)~1,
  (HighPCO2>75)~3
)))

#Glucose
prism$nonneuro.glucose <- with(prism,ifelse(GlucoseND == "Not Done" | HighGlucose<=200,0,2))

#Potassium
prism$nonneuro.potassium <- with(prism,ifelse(PotassiumND == "Not Done" | HighPotassium<=6.9,0,3))

#Creatinine
prism <- prism %>% mutate(.,nonneuro.cr = with(.,case_when(
  ((age == "0 to <14 days" | age == "14 days to <1 month") & HighCreatinine<=0.85)~0,
  ((age == "0 to <14 days" | age == "14 days to <1 month") & HighCreatinine>0.85)~2,
  (age == "1 month to <12 months" & HighCreatinine<=0.9)~0,
  (age == "1 month to <12 months" & HighCreatinine>0.9)~2,
  (age == ">12 months to 144 months" & HighCreatinine<=0.9)~0,
  (age == ">12 months to 144 months" & HighCreatinine>0.9)~2,
  (age == ">144 months" & HighCreatinine<=1.3)~0,
  (age == ">144 months" & HighCreatinine>1.3)~2,
)))
prism$nonneuro.cr <- with(prism,ifelse(is.na(nonneuro.cr),0,nonneuro.cr))

#BUN
prism <- prism %>% mutate(.,nonneuro.bun = with(.,case_when(
  ((age == "0 to <14 days" | age == "14 days to <1 month") & HighBUN<=11.9)~0,
  ((age == "0 to <14 days" | age == "14 days to <1 month") & HighBUN>11.9)~3,
  ((age == "1 month to <12 months" | age == ">12 months to 144 months" |
      age == ">144 months") & HighBUN<=14.9)~0,
  ((age == "1 month to <12 months" | age == ">12 months to 144 months" |
      age == ">144 months") & HighBUN>14.9)~3
)))
prism$nonneuro.bun <- with(prism,ifelse(is.na(nonneuro.bun),0,nonneuro.bun))

#WBC
prism$nonneuro.wbc <- with(prism,ifelse(WBCND == "Not Done" | LowWBC>=3,0,4))

#PT or PTT
prism <- prism %>% mutate(.,nonneuro.ptnt = with(.,case_when(
    ((age == "0 to <14 days" | age == "14 days to <1 month") & (HighPT>22 | HighPTT>85))~3,
  ((age == "1 month to <12 months" | age == ">12 months to 144 months" |
      age == ">144 months") & (HighPT>22 | HighPTT>57))~3
)))
prism$nonneuro.ptnt <- with(prism,ifelse(is.na(nonneuro.ptnt),0,nonneuro.ptnt))

#Platelets
prism <- prism %>% mutate(.,nonneuro.plts = with(.,case_when(
  (LowPlatelets>=100 & LowPlatelets<=200)~2,
  (LowPlatelets>=50 & LowPlatelets<=99)~4,
  (LowPlatelets<50)~5
)))
prism$nonneuro.plts <- with(prism,ifelse(is.na(nonneuro.plts),0,nonneuro.plts))

#Assign non-neurologic score
prism$nonneurologic.score <- rowSums(prism[,c("nonneuro.sbp","nonneuro.hr","nonneuro.temp",
                                              "nonneuro.acidosis","nonneuro.totalco2",
                                              "nonneuro.ph","nonneuro.pao2","nonneuro.pco2",
                                              "nonneuro.glucose","nonneuro.potassium","nonneuro.cr",
                                              "nonneuro.bun","nonneuro.wbc","nonneuro.ptnt","nonneuro.plts")])

#GCS
prism$neuro.gcs <- with(prism,ifelse((as.numeric(GCSWorstTotal)<8 | LOCWorst == "Coma (unresponsive)")
                                     & CNSInjury == "Yes",5,0))
prism$neuro.gcs <- with(prism,ifelse(is.na(neuro.gcs),0,neuro.gcs))

#Pupils
prism$neuro.pupils <- with(prism,ifelse(PupilWorst == "One non-reactive (>3 mm)",7,
                                        ifelse(PupilWorst == "Both non-reactive (> 3mm)",11,0)))
prism$neuro.pupils <- with(prism,ifelse(is.na(neuro.pupils),0,neuro.pupils))

#Assign neurologic score
prism$neurologic.score <- rowSums(prism[,c("neuro.gcs","neuro.pupils")])

#Age for trichotomous model
prism$age0to14d <- with(prism,ifelse(age == "0 to <14 days",1,0))
prism$age14dto1m <- with(prism,ifelse(age == "14 days to <1 month",1,0))
prism$age1to12m <- with(prism,ifelse(age == "1 month to <12 months",1,0))
prism$ageg12m <- with(prism,ifelse(age == ">12 months to 144 months" |
                                     age == ">144 months",1,0))

prism$agecat <- with(prism,ifelse(age0to14d == 1, "0 to <14 days",
                                  ifelse(age14dto1m == 1, "14 days to <1 month",
                                         ifelse(age1to12m == 1, "1 month to <12 months",
                                                ifelse(ageg12m == 1, ">12 months","")))))

#Admission source
##Outside Hospital
osh <- subset(picuadmit, PICUAdmitSource == "Direct admission from outside of the study hospital")
prism$osh <- 0
prism$osh[prism$PudID %in% osh$PudID] <- 1

##Inpatient
inpt <- subset(picuadmit, PICUAdmitSource == "Study hospital general care floor" |
                 PICUAdmitSource == "Study hospital intermediate care unit" |
                 PICUAdmitSource == "Study hospital monitoring unit" |
                 PICUAdmitSource == "Study hospital other ICU" |
                 PICUAdmitSource == "Study hospital other location")
prism$inpt <- 0
prism$inpt[prism$PudID %in% inpt$PudID] <- 1

##Emergency Dept
ed <- subset(picuadmit, PICUAdmitSource == "Study hospital emergency department")
prism$ed <- 0
prism$ed[prism$PudID %in% ed$PudID] <- 1

##OR/PACU
or <- subset(picuadmit, PICUAdmitSource == "Study hospital operating room")
prism$or <- 0
prism$or[prism$PudID %in% or$PudID] <- 1

prism$admitsourcecat <- with(prism, ifelse(osh == 1, "Direct: Referral Hosp",
                                           ifelse(inpt == 1, "Inpatient Unit",
                                                  ifelse(ed == 1, "ED",
                                                         ifelse(or == 1, "OR/PACU","")))))

#Cardiac arrest
cpr_before <- subset(ce,BaselineCEDesc == "Cardiac Arrest" & 
                        (BaselineCEInterval == "< 12 hours" | BaselineCEInterval == "12 to < 24 hours"))
picutime <- picuadmit[,c("PudID","PICUAdmitTime")]
hospce <- left_join(hospce,picutime,by="PudID")
hospce$diff <- with(hospce, HospCETime - PICUAdmitTime)
hospcpr <- subset(hospce, HospCEDesc == "Cardiac Arrest" & diff<0 & HOSPCEDAY == 0)

cpr_1 <- hospcpr[,c(1)]
cpr_2 <- cpr_before[,c(1)]
cpr_1 <- as.data.frame(cpr_1)
cpr_2 <- as.data.frame(cpr_2)
names(cpr_1) <- c("PudID")
names(cpr_2) <- c("PudID")
cpr <- rbind(cpr_1,cpr_2)
cpr <- unique(cpr)

prism$cpr <- 0
prism$cpr[prism$PudID %in% cpr$PudID] <- 1

#Acute (Secondary Dx) or Chronic Cancer
picusecondcancer <- subset(picuseconddx,PICUAdmitSecondDxCat == "Cancer")
picuchronicancer <- subset(picuchrondx, PICUAdmitChronicDxCat == "Cancer")
secondcancerids <- picusecondcancer[,c("PudID")]
chroniccancerids <- picuchronicancer[,c("PudID")]
secondcancerids <- as.data.frame(secondcancerids)
chroniccancerids <- as.data.frame(chroniccancerids)
names(secondcancerids) <- c("PudID")
names(chroniccancerids) <- c("PudID")

cancerids <- rbind(secondcancerids, chroniccancerids)
cancerids <- unique(cancerids)

prism$cancer <- 0
prism$cancer[prism$PudID %in% cancerids$PudID] <- 1

#Trauma
trauma <- subset(picuadmit, PICUPostOpType == "Trauma" | PICUAdmitClinicalService == "Trauma surgery" |
                   PICUAdmitPrimaryDx == "Trauma" | PICUAdmitPrimaryDx == "Drowning / asphyxia / hanging")
prism$trauma <- 0
prism$trauma[prism$PudID %in% trauma$PudID] <- 1

#Primary system of dysfunction
cvresp <- subset(picuadmit, PICUAdmitPrimaryDx == "Airway/tracheal abnormality, obstruction, surgery" |
                   PICUAdmitPrimaryDx == "Asthma" | PICUAdmitPrimaryDx == "Cardiovascular disease - acquired" |
                   PICUAdmitPrimaryDx == "Cardiovascular disease - arrhythmia" | 
                   PICUAdmitPrimaryDx == "Cardiovascular disease - congenital" |
                   PICUAdmitPrimaryDx == "Pertussis" | PICUAdmitPrimaryDx == "Respiratory distress / failure" |
                   PICUAdmitPrimaryDx == "Cardiac arrest")
cancerprimary <- subset(picuadmit, PICUAdmitPrimaryDx == "Cancer")
lowrisk <- subset(picuadmit, PICUAdmitPrimaryDx == "Diabetic ketoacidosis (DKA)" |
                    PICUAdmitPrimaryDx == "Hematologic disorder" |
                    PICUAdmitPrimaryDx == "Musculoskeletal condition" |
                    PICUAdmitPrimaryDx == "Scoliosis / spine surgery" |
                    PICUAdmitPrimaryDx == "Renal failure")
neurologic <- subset(picuadmit, PICUAdmitPrimaryDx == "Central nervous system infection" |
                       PICUAdmitPrimaryDx == "Neurological - cords, bones" |
                       PICUAdmitPrimaryDx == "Neurological - vascular malformations" |
                       PICUAdmitPrimaryDx == "Neurological CSF related (hydrocephalus / Chiari / fenestrations / arachnoid cysts)" |
                       PICUAdmitPrimaryDx == "Neurological miscellaneous" |
                       PICUAdmitPrimaryDx == "Seizures" |
                       PICUAdmitPrimaryDx == "Stroke / Cerebral Ischemia / Cerebral infarction")
otherprimary <- subset(picuadmit, PICUAdmitPrimaryDx == "Congenital anomaly or chromosomal defect" |
                         PICUAdmitPrimaryDx == "Gastrointestinal disorder" |
                         PICUAdmitPrimaryDx == "Ingestion (drug or toxin)" |
                         PICUAdmitPrimaryDx == "Other miscellaneous" |
                         PICUAdmitPrimaryDx == "Sepsis / SIRS / Septic shock" |
                         PICUAdmitPrimaryDx == "Transplant" |
                         PICUAdmitPrimaryDx == "Trauma" |
                         PICUAdmitPrimaryDx == "Drowning / asphyxia / hanging")

prism$cvresp <- 0
prism$cvresp[prism$PudID %in% cvresp$PudID] <- 1
prism$cancerprimary <- 0
prism$cancerprimary[prism$PudID %in% cancerprimary$PudID] <- 1
prism$lowrisk <- 0
prism$lowrisk[prism$PudID %in% lowrisk$PudID] <- 1
prism$neurologic <- 0
prism$neurologic[prism$PudID %in% neurologic$PudID] <- 1
prism$otherprimary <- 0
prism$otherprimary[prism$PudID %in% otherprimary$PudID] <- 1

prism$primarysyscat <- with(prism, ifelse(cvresp == 1, "CV/Resp",
                                          ifelse(cancerprimary == 1, "Cancer",
                                                 ifelse(lowrisk == 1, "Low Risk",
                                                        ifelse(neurologic == 1, "Neurologic",
                                                               ifelse(otherprimary == 1, "Other",""))))))

#FSS
FSSdc_num <- lapply(FSSdc, function(i)
  ifelse(i == "Normal",1,
         ifelse(i == "Mild dysfunction",2,
                ifelse(i == "Moderate dysfunction",3,
                       ifelse(i == "Severe dysfunction",4,
                              ifelse(i == "Very severe dysfunction",5,
                                     ifelse(i == "",0,i)))))) #changed i to 0, invalid FSS if blank
)
FSSdc_num <- as.data.frame(lapply(FSSdc_num,function(i) as.numeric(i)))

FSSdc_num$FSSdcsum <- rowSums(FSSdc_num[,c("HospDisFSSMental","HospDisFSSSensory","HospDisFSSCommun",
                                                 "HospDisFSSMotor","HospDisFSSFeeding","HospDisFSSResp")])

FSSall <- left_join(FSSbase_num,FSSdc_num,by="PudID")

FSSall$basedcchange <- with(FSSall,FSSdcsum - FSSbasesum)

prism <- left_join(prism, FSSall, by="PudID")

prism$FSSgood <- with(prism,ifelse(FSSbasesum == 6 | FSSbasesum == 7,1,0))

#Mortality
death <- hospdc[,c("PudID","HospDisAlive")]
death$mortality <- with(death,ifelse(HospDisAlive == "No",1,0))

prism <- left_join(prism, death, by="PudID")

#Morbidity
prism$morbidity <- with(prism,ifelse(basedcchange>=3,1,0))

#Construct clean prism table
prism_table_topicc <- select(prism, c(PudID, agecat, admitsourcecat, cpr, cancer, trauma, primarysyscat,
                                      FSSgood, neurologic.score, nonneurologic.score, mortality, morbidity))

prism_table_topicc$agecat <- with(prism_table_topicc,factor(agecat, levels = c(">12 months",
                                                                               "1 month to <12 months",
                                                                               "14 days to <1 month",
                                                                               "0 to <14 days")))

prism_table_topicc$admitsourcecat <- with(prism_table_topicc,factor(admitsourcecat, levels=c("OR/PACU", "ED", 
                                                          "Inpatient Unit","Direct: Referral Hosp")))

prism_table_topicc$primarysyscat <- with(prism_table_topicc, factor(primarysyscat, 
                                                                    levels = c("CV/Resp",
                                                                               "Cancer",
                                                                               "Low Risk",
                                                                               "Neurologic",
                                                                               "Other")))

setwd("/home/repositories/git/nkmj/topicc/tables")
saveRDS(prism_table_topicc, file="prism_topicc.rds")
write.csv(prism_table_topicc,"prism_topicc.csv",row.names = FALSE)

prism_table_topicc$morbmort = prism_table_topicc$morbidity + prism_table_topicc$mortality

prism_model_test <- glm(mortality ~ agecat + admitsourcecat + cpr + cancer + trauma + primarysyscat +
                          FSSgood + neurologic.score + nonneurologic.score, data = prism_table_topicc, 
                        family = binomial(link = "logit"))

neuroprism <- prism_table_topicc[which(prism$CNSInjury=="Yes"),names(prism_table_topicc)]

nprism_model <- glm(mortality ~ agecat + admitsourcecat + cpr + cancer + trauma + primarysyscat +
                                       FSSgood + neurologic.score + nonneurologic.score, data = neuroprism, 
                                     family = binomial(link = "logit"))
npredict <- predict(nprism_model, type = 'response')
fg <- npredict[neuroprism$mortality==1]
bg <- npredict[neuroprism$mortality==0]
#fg <- npredict[neuroprism$morbmort==1]
#bg <- npredict[neuroprism$morbmort==0]
plot(roc.curve(fg,bg,curve=T))
plot(pr.curve(fg,bg,curve=T))

set.seed(42)
dt = sort(sample(nrow(neuroprism),nrow(neuroprism)*0.7))
neuroprism_train <- neuroprism[dt,]
neuroprism_test<- neuroprism[-dt,]

nprism_tt_model <- glm(mortality ~ agecat + admitsourcecat + cpr + cancer + trauma + primarysyscat +
                                       FSSgood + neurologic.score + nonneurologic.score, data = neuroprism_train, 
                                     family = binomial(link = "logit"))
npredict_tt <- predict(nprism_tt_model,neuroprism_test, type = 'response')
prismpredict_tt <- predict(prism_model_test,neuroprism_test, type = 'response')
fg <- npredict_tt[neuroprism_test$mortality==1]
bg <- npredict_tt[neuroprism_test$mortality==0]
fg <- prismpredict_tt[neuroprism_test$mortality==1]
bg <- prismpredict_tt[neuroprism_test$mortality==0]
#fg <- npredict[neuroprism$morbmort==1]
#bg <- npredict[neuroprism$morbmort==0]
plot(roc.curve(fg,bg,curve=T))
plot(pr.curve(fg,bg,curve=T))

rocs = c()
prcs = c()
for (i in 1:20) {
    set.seed(i)
    dt = sort(sample(nrow(neuroprism),nrow(neuroprism)*0.7))
    neuroprism_train <- neuroprism[dt,]
    neuroprism_test<- neuroprism[-dt,]
    
    nprism_tt_model <- glm(mortality ~ agecat + admitsourcecat + cpr + cancer + trauma + primarysyscat +
                                           FSSgood + neurologic.score + nonneurologic.score, data = neuroprism_train, 
                                         family = binomial(link = "logit"))
    npredict_tt <- predict(nprism_tt_model,neuroprism_test, type = 'response')
    fg <- npredict_tt[neuroprism_test$mortality==1]
    bg <- npredict_tt[neuroprism_test$mortality==0]
    rocs = c(rocs,roc.curve(fg,bg)$auc)
    prcs = c(prcs,pr.curve(fg,bg)$auc.integral)
}
sort(rocs)
mean(rocs)
sort(prcs)
mean(prcs)