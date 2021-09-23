#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from sys import exit
from osgeo import gdal,gdalconst
import matplotlib.pyplot as plt
from numba import njit
from numba import jit
import os
import xarray
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = 20, 12


# In[2]:


#BORDERS
countries=["United States of America","Spain"]

def get_borders(country):
    borders=gdal.Open("./data/national_grids/gpw_v4_national_identifier_grid_rev11_30_min.tif").ReadAsArray()
    reference=pd.read_excel("./data/national_grids/gpw-v4-documentation-rev11/gpw-v4-country-level-summary-rev11.xlsx",
                            sheet_name="GPWv4 Rev11 Summary",skiprows=1)
    reference=reference.loc[reference.iloc[:,0]==country,"ISO Numeric"].values[0]
    return (borders==reference).astype("int8")


# In[3]:


#Population version: 

###
# isimip2b.input_secondary.population.ssp2soc.population.yearly
# Description: Population data set prepared for ISIMIP2b. Data Set Details: https://www.isimip.org/gett..., ISIMIP Terms of Use: https://www.isimip.org/prot..., Published under CC BY 4.0 licence.
# Data Node: esg.pik-potsdam.de
# Version: 20200716
# http://esg.pik-potsdam.de/thredds/fileServer/isimip_dataroot_2/isimip2b/input_secondary/population/ssp2soc/population/yearly/v20200716/population_ssp2soc_0p5deg_annual_2006-2100.nc
##

def get_population(year):
    
    if (year<2006)|(year>2100):
        print("Year out of range. Only the period 2006-2100 is available")
        exit
    else:
        return gdal.Open("./data/population/population_ssp2soc_0p5deg_annual_2006-2100.nc").ReadAsArray()[(year-2006),:,:]


# In[4]:


un_translate={"Czech Republic":"Czechia",
                 'United Kingdom of Great Britain and Northern Ireland':"United Kingdom",
             "Cape Verde":"Cabo Verde",
             "Micronesia (Federated States of)":"Micronesia (Fed. States of)",
             "The former Yugoslav Republic of Macedonia":"North Macedonia",
             "Democratic People's Republic of Korea":"Dem. People's Republic of Korea",
             "Taiwan":"China, Taiwan Province of China","Western Samoa":"Samoa"
}
    
def get_proportions(country,year=2015):
    if year==2016:
        year=2015
    if year<=2020:
        un=pd.read_excel("./data/population/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx",sheet_name="ESTIMATES",
                        skiprows=16)
    else:
        un=pd.read_excel("./data/population/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx",sheet_name="HIGH VARIANT",
                        skiprows=16)
    un=un[un["Reference date (as of 1 July)"]==year]

    
    if country in un_translate.keys():
        country=un_translate[country]
    
    df=un[un["Region, subregion, country or area *"]==country]
    
    df["95+"]=df["95-99"]+df["100+"]
    df.drop(["95-99","100+"],axis=1,inplace=True)
    df=df.iloc[:,8:] #Getting ages
    df=df.iloc[0,:]/df.sum(axis=1).values #Calculating proportions
    return df.values    


# In[5]:


gbd_translate={"Iran":"Iran (Islamic Republic of)",
                  "The Gambia":"Gambia","Cote d'Ivoire":"Côte d'Ivoire","Tanzania":"United Republic of Tanzania",
                  "Moldova":"Republic of Moldova","Vietnam":"Viet Nam","Brunei":"Brunei Darussalam",
                  "The Bahamas":"Bahamas","South Korea":"Republic of Korea",
                  "Venezuela":"Venezuela (Bolivarian Republic of)","Syria":"Syrian Arab Republic",
                  "Laos":"Lao People's Democratic Republic","Bolivia":"Bolivia (Plurinational State of)",
               "United States":"United States of America",
              "Palestine":"State of Palestine",
              "United Kingdom":"United Kingdom of Great Britain and Northern Ireland",
               "Virgin Islands, U.S.":"United States Virgin Islands",
               "Federated States of Micronesia":"Micronesia (Federated States of)",
               "Macedonia":"The former Yugoslav Republic of Macedonia",
               "North Korea":"Democratic People's Republic of Korea",
               "Taiwan (Province of China)":"Taiwan",
               "Samoa":"Western Samoa"
               
              }
gbd_translate= {v: k for k, v in gbd_translate.items()}

causes_rr=["LRI","COPD","IHD","STROKE","LC","ALL"]


def get_mort(country,year):
    ind={"LC":"LC_ALL_RISKS_MORT","IHD":"IHD_ALL_RISKS_MORT_AGEGR","STROKE":"STROKE_ALL_RISKS_MORT_AGEGR",
    "LRI":"LRI_ALL_RISKS_MORT","COPD":"COPD_ALL_RISKS_MORT"}
    results={}
    if country in gbd_translate.keys():
        country=gbd_translate[country]
    for cause in causes_rr[:-1]:
        rita_file=pd.read_excel("./data/GBD_data/MORTALITY_RATES_COUNTRIES_PROJ_GBD2017_6COD.xlsx",sheet_name=ind[cause])
        if (year==2010)|(year==2016):
            rita_file=rita_file[(rita_file.location_name==country)&(rita_file.year==2015)]
        elif year==2050:
            rita_file=rita_file[(rita_file.location_name==country)&(rita_file.year==2040)]
        else:
            rita_file=rita_file[(rita_file.location_name==country)&(rita_file.year==year)]
        results[cause]=rita_file.val.values
        
    return results    
get_mort("United States of America",2016)


# In[6]:


def get_coefficients(cause_rr):
    rr=pd.read_csv("./data/GBD_data/RR_PARAMS_GBD2018.csv",sep=",")
    cause=rr[rr["COD"]==cause_rr]
    return cause["alpha_med"].values,cause["beta_med"].values,cause["delta_med"].values,cause["zcf_med"].values

@njit
def get_rr_array(alpha,beta,delta,zcf,conc_array,cause):
    a=conc_array.shape[0]
    b=conc_array.shape[1]
    c=len(alpha)
    rr_array=np.ones((c,a,b))
    for k in range(c):
        for i in range(a):
            for j in range(b):
                if conc_array[i,j]>zcf[k]:    
                    rr_array[k,i,j]=1+(alpha[k]*(1-np.exp(-beta[k]*(conc_array[i,j]-zcf[k])**delta[k])))               
    return rr_array
    
    
#@njit
def process_array(population_array,conc_array,proportions,mort_rate,cause,rr_array):

    results=conc_array*0.

    for i in range(population_array.shape[0]):
        for j in range(population_array.shape[1]):
            if (population_array[i,j]==0.):
                results[i,j]=0.
            else:
                if cause in ["IHD","STROKE"]:
                    atr_mortality=mort_rate[-15:]*(rr_array[:,i,j]-1)
                    mort=np.dot(population_array[i,j]*proportions[-15:]/100000,atr_mortality)

                else:

                    atr_mortality=mort_rate*(rr_array[:,i,j]-1)
                    mort=population_array[i,j]*atr_mortality/100000
                results[i,j]=mort
                
    return results

@njit
def population_weighted_conc(population_array,conc_array):
    results=0.
    pop=0
    for i in range(population_array.shape[0]):
        for j in range(population_array.shape[1]):
            if (conc_array[i,j]>=0) & (population_array[i,j]>0):
                results+=(population_array[i,j]*conc_array[i,j])
                pop+=population_array[i,j]
    return results/pop



# ## MARIANNE PM25 AND SALT FROM AMAP

# In[18]:




salt=xarray.open_dataset("./data/model_layers/pm25_sea_salt_dust_amap.nc")
print(salt)


# In[ ]:



    


# In[9]:


model_dir="./data/model_layers/corrected/"

year=2016
scenario=""
model=salt["pm25"].data

files=sorted([x for x in os.listdir(model_dir) if (str(year) in x)&(scenario in x)&("speciated" not in x)])

files


# In[10]:


plt.subplots(figsize=(20,12))
plt.imshow(np.flipud(model),cmap="Spectral_r")
plt.axis("off")
plt.colorbar(extend="both",orientation="horizontal",aspect=50,pad=-0.1)


# Estimation using AMAP salt layer

# In[7]:


def save_raster(array,raster_template,output_name,fill=False):

    ds = gdal.Open(raster_template)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_name, rows, cols, 1, gdal.GDT_Float32)
    outdata.GetRasterBand(1).SetNoDataValue(-9999)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(array)
    if fill:
        result = gdal.FillNodata(targetBand =outdata.GetRasterBand(1), maskBand = None, 
                     maxSearchDist = 50, smoothingIterations = 2)

    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band=None
    ds=None


# In[8]:



#scenarios=[""]
#years=[2010]



def get_conc_array(year=2015,scenario="CLE"):
    amapdir="/home/gaia/Documents/AMAP/repo/data/Model_data/PM25/downscaled/"
    salt=xarray.open_dataset("./data/model_layers/pm25_sea_salt_dust_amap.nc")

    model_dir="./data/model_layers/corrected/"

    model=0.*salt["pm25"].data

    files=sorted([x for x in os.listdir(model_dir) if (str(year) in x)&(scenario in x)&("xml" not in x)])
    variables=["BC","NO3","OA","POA","SO4"]
    for i in range(len(variables)):
        
        model=model+xarray.open_dataset(os.path.join(model_dir,files[i]),engine="netcdf4")[variables[i]].data
    return(np.flipud(model))

def process_all(year,scenario):
    country="China"
    #c -> country class
    
    

        
        
    #         year_band={2010:4,2030:6,2050:8}
    #         population_file="./data/SSP2_POP_GRID_0.5x0.5.tif"
    population_raster=get_borders(country)*get_population(year)
    proportions=get_proportions(country,year)

    print("Total population ",np.sum(population_raster))
    print("Done")

    mortality=get_mort(country,year)
    

    yearidx={2016:1,2030:2,2050:3}
    
    model_raster=get_conc_array(year=year,scenario=scenario)


    health_impacts={}
    age_impacts={}
    rr_arrays={}
    for cause in ["COPD","LRI","LC","IHD","STROKE"]:

        alpha,beta,delta,zcf=get_coefficients(cause)

        rr_array=get_rr_array(alpha,beta,delta,zcf,model_raster,cause)
        rr_arrays[cause]=rr_array
        health_impacts[cause]=process_array(population_raster,model_raster,proportions,
                                         mortality[cause],cause,rr_array)

   
    weighted_conc=population_weighted_conc(population_raster,model_raster)
    results=[rr_arrays,weighted_conc,health_impacts]
    
    return results


# Estimation using PM25 from OsloCTM

# In[9]:


get_conc_array(year=2016,scenario="")


# In[10]:


_,pwc,impacts=process_all(2016,"")


# In[22]:


scenarios=["","CLE","MFR","CLE","MFR"]
years=[2016,2030,2030,2050,2050]
results=pd.DataFrame()
for  i in range(len(scenarios)):
    scenario=scenarios[i]
    year=years[i]
    _,pwc,impacts=process_all(year,scenario)
    total=impacts["COPD"]+impacts["LRI"]+impacts["LC"]+impacts["STROKE"]+impacts["IHD"]
    save_raster(total,"/home/gaia/Documents/AMAP/repo/data/EMEP_Average_2014_2015.tif","./impacts_v2/impacts_"+str(year)+scenario+"_nodust.tif")
    print("Finished ok")


# In[23]:


scenarios=["","CLE","MFR","CLE","MFR"]
years=[2016,2030,2030,2050,2050]
results=pd.DataFrame()
for  i in range(len(scenarios)):
    scenario=scenarios[i]
    year=years[i]
    _,pwc,impacts=process_all(year,scenario)
    results=results.append({"country":"China","year":year,"scenario":scenario,"pwc":pwc,"COPD":np.sum(impacts["COPD"]),
                       "LRI":np.sum(impacts["LRI"]),"LC":np.sum(impacts["LC"]),"IHD":np.sum(impacts["IHD"]),
                       "STROKE":np.sum(impacts["STROKE"])},ignore_index=True)
    print("Finished ok")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


results.to_csv(".first_results_china_v2_nodust.csv")


# In[ ]:


from pyresample.bilinear import NumpyBilinearResampler
from pyresample.geometry import SwathDefinition


def_b = SwathDefinition(lons=z.variables["lon"], lats=z.variables["lat"])

res= NumpyBilinearResampler(def_b, area_def,radius_of_influence=50000)
results=res.resample(z.variables["TX"].data[0,:,:])


# In[ ]:





# In[54]:


gdal.Open("./data/paper/pm25_downscaled_CESM_2015.nc").ReadAsArray()


# In[55]:


def save_raster(array,out_transform,out_meta,output_fname):
    out_meta.update({"driver": "GTiff",
                 "height": array.shape[1],
                 "width": array.shape[2],
                 "transform": out_transform})

    with rasterio.open(output_fname, "w", **out_meta) as dest:
        dest.write(array)

def process_all(country,year,model_file=gdal.Open("./data/paper/pm25_downscaled_emulator_SLCF_total.nc").GetSubDatasets()[3][0],output=False):
    #c -> country class
    
    

        
        
    #         year_band={2010:4,2030:6,2050:8}
    #         population_file="./data/SSP2_POP_GRID_0.5x0.5.tif"
    population_raster=get_borders(country)*get_population(year)
    proportions=get_proportions(country,year)

    print("Total population ",np.sum(population_raster))
    print("Done")

    mortality=get_mort(country,year)
    

    yearidx={2015:1,2030:2,2050:3}
    
    model_array=gdal.Open(model_file).ReadAsArray()[yearidx[year],:,:]
    print(model_array.shape)
    
    dust_array=gdal.Open("./data/pm25_salt.nc").ReadAsArray()


    model_raster=(model_array+dust_array)

    print("Done")


    health_impacts={}
    age_impacts={}
    rr_arrays={}
    for cause in ["COPD","LRI","LC","IHD","STROKE"]:

        alpha,beta,delta,zcf=get_coefficients(cause)

        rr_array=get_rr_array(alpha,beta,delta,zcf,model_raster,cause)
        rr_arrays[cause]=rr_array
        health_impacts[cause]=process_array(population_raster,model_raster,proportions,
                                         mortality[cause],cause,rr_array)

    if output:
        save_raster(np.expand_dims(health_impacts,0),model_transform,model_meta,os.path.join(os.getcwd(),results,country_name+"_"+cause+"_"+str(year)+".tiff"))

    weighted_conc=population_weighted_conc(population_raster,model_raster)
    results=[rr_arrays,weighted_conc,health_impacts]
    
    return results


    


# In[16]:



with open("./data/non_included_countries.txt", "r") as file:
    non_included_countries = eval(file.readline())
reference=pd.read_excel("./data/national_grids/gpw-v4-documentation-rev11/gpw-v4-country-level-summary-rev11.xlsx",
                       sheet_name="GPWv4 Rev11 Summary",skiprows=1)
countries=reference["Country or Territory Name"].unique()
countries=[x for x in countries if x not in non_included_countries]
reference=None


# In[59]:


results=pd.DataFrame()
check_these=[]

for country in countries:
    print(country)
    try:
        _,pwc,impacts=process_all(country,2050)
        results=results.append({"country":country,"year":2050,"scenario":"SLCF","pwc":pwc,"COPD":np.sum(impacts["COPD"]),
                       "LRI":np.sum(impacts["LRI"]),"LC":np.sum(impacts["LC"]),"IHD":np.sum(impacts["IHD"]),
                       "STROKE":np.sum(impacts["STROKE"])},ignore_index=True)
        print("Finished ok")
    except:
        check_these.append(country)
    
    
    


# In[30]:


z=process_all("Spain",year=2030)


# In[58]:


results


# In[60]:


#results.to_csv("./2050_SLCF_impacts.csv",index=False)


# In[46]:


results


# In[64]:


# reference=pd.read_excel("./data/borders/gpw-v4-documentation-rev11/gpw-v4-country-level-summary-rev11.xlsx",
#                        sheet_name="GPWv4 Rev11 Summary",skiprows=1)
# countries_masks=reference["Country or Territory Name"].unique()

# countries_inc=gbd=pd.read_csv("./data/GBD/IHME-GBD_2017_DATA-23a76344-1.csv").location_name.unique()

# missing=[x for x in countries_masks if (x not in countries_inc)&(x not in gbd_translate.keys())]
# missing=missing+["Andorra","American Samoa","Bermuda","Dominica","Greenland",
#                 'Marshall Islands', 'Northern Mariana Islands', 'Swaziland']

# countries_un=pd.read_excel("./data/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx",sheet_name="ESTIMATES",
#                          skiprows=16)
# countries_un=np.unique(countries_un.iloc[:,2])
# countries=[x for x in countries_masks if (x not in countries_un)&(x not in un_translate.keys())&(x not in missing)]
# with open("./data/non_included_countries.txt", "w") as file:
#     file.write(str(missing))


# In[12]:


countries


# In[9]:


scenarios=["","CLE","MFR","CLE","MFR"]
years=[2015,2030,2030,2050,2050]
#scenarios=[""]
#years=[2010]


def get_nox_array(year=2015,scenario="CLE"):
    noxfolder="/media/gaia/Seagate Expansion Drive/Liliana/fon/Downscaling/averaged_susan"

    if year!=2015:
        filename=[x for x in os.listdir(noxfolder) if (str(year) in x)&(scenario in x)][0]
    else:
        filename=[x for x in os.listdir(noxfolder) if (str(year) in x)][0]
    conc_array=gdal.Open(os.path.join(noxfolder,filename)).ReadAsArray()

    return conc_array

results=pd.DataFrame()
for i in range(len(years)):
    year=years[i]
    print(year)
    scenario=scenarios[i]
    print(scenario)
    conc_array=get_nox_array(year=year,scenario=scenario)
    for country in countries:


        try:
            print("Country.... ",country)
            population=get_population(country,year=year)
            print("Population -> ",np.sum(population))
            mask=np.where(conc_array>=0)
            pwc=np.sum(population[mask]*conc_array[mask])/np.sum(population[mask])
            
            
            
            
            print("pwc = ",pwc)
           
            results=results.append({"country":country,"year":year,"scenario":scenario,"pwc":pwc},ignore_index=True)
        except:
            results=results.append({"country":country,"year":"","scenario":"","pwc":""},ignore_index=True)
    


# In[21]:


mid=np.log(1-0.26)/-10
low=np.log(1-0.10)/-10
high=np.log(1-0.37)/-10

@njit
def get_rr_array(conc_array,beta=low):
    a=conc_array.shape[0]
    b=conc_array.shape[1]
    rr_array=np.zeros((a,b),dtype=np.float32)
    for i in range(a):
        for j in range(b):
            if conc_array[i,j]>5:
                rr_array[i,j]=(1-np.exp(-beta*conc_array[i,j]))
    
    return rr_array


@njit
def estimate_impact(rr_array,population_array,proportion,incidence):

    a=rr_array.shape[0]
    b=rr_array.shape[1]
    results=0
    population=0
    for i in range(a):
        for j in range(b):
            results=results+(proportion*incidence*population_array[i,j]*rr_array[i,j])/100000
            population+=population_array[i,j]*proportion
    return results,population

################
################

def get_nox_array(year=2015,scenario="CLE"):
    noxfolder="/media/gaia/Seagate Expansion Drive/Liliana/fon/Downscaling/averaged_susan"

    if year!=2015:
        filename=[x for x in os.listdir(noxfolder) if (str(year) in x)&(scenario in x)][0]
    else:
        filename=[x for x in os.listdir(noxfolder) if (str(year) in x)][0]
    conc_array=gdal.Open(os.path.join(noxfolder,filename)).ReadAsArray()

    return conc_array

scenarios=["CLE","MFR","CLE","MFR"]
years=[2030,2030,2050,2050]
#scenarios=[""]
#years=[2010]

results=pd.DataFrame()
for i in range(len(years)):
    year=years[i]
    print(year)
    scenario=scenarios[i]
    print(scenario)
    conc_array=get_nox_array(year=year,scenario=scenario)
    rr_array=get_rr_array(conc_array)
    conc_array=None

    for country in countries:


        try:
            print("Country.... ",country)
            population=get_population(country,year=year)
            
            print("Population -> ",np.sum(population))
            proportions=get_proportions(country,year)
            print("Proportions -> ",proportions)
            incidence=get_incidence(country,2015)[0] #Only using the mid value!
            print("Incidence -> ",incidence)
            total=0
            children=0
            
            for agegroup in range(len(proportions)):
                tot,popul=estimate_impact(rr_array,population,proportions[agegroup],incidence[agegroup])    
                children+=popul
                total+=tot
            
            
            print("Number of cases caused ",total)
            print("Number of cases inc/100.000 ",(100000)*total/children)
            print("Population under up until 18y ",children)
            results=results.append({"country":country,"year":year,"scenario":scenario,"1-18 population":children,"NO2cases":total,"NO2incidence":100000*total/children},ignore_index=True)
        except:
            results=results.append({"country":country,"year":"","scenario":"","1-18 population":"","NO2cases":"","NO2incidence":""},ignore_index=True)
    rr_array=None


# In[20]:


#results.to_csv("./pwc_mediumpop.csv")
results.to_csv("./asthma_highvariant_lowfunction.csv",index=False)
#results=pd.read_csv("./asthma_estimates_v2.csv")
#results=results.iloc[:,1:]
#results.head()


# In[15]:





# In[43]:


np.unique(results["country"][results.iloc[:,0]==""])


# In[18]:


# conc_raster=gdal.Open("./data/nox/Exposure_2011_Susan_extended.tif")
# trans=conc_raster.GetGeoTransform()
# projection=conc_raster.GetProjection()

# conc_array=(100*conc_raster.ReadAsArray())
# conc_array[conc_array<0]=-10
# rows,cols=conc_array.shape

# # create the output image
# driver = gdal.GetDriverByName( 'GTiff' )
# #print driver
# outDs = driver.Create("./data/nox/Exposure_2011_Susan_extended_int.tif", cols, rows, 1, gdalconst.GDT_Int16)


# outBand = outDs.GetRasterBand(1)

# # write the data
# outBand.WriteArray(conc_array)

# # flush data to disk, set the NoData value and calculate stats
# outBand.FlushCache()
# outBand.SetNoDataValue(-10)

# # georeference the image and set the projection
# outDs.SetGeoTransform(trans)
# outDs.SetProjection(projection)
# outDs = None
# del conc_array
# conc_raster=None


# In[ ]:





# In[9]:


paper_data=pd.read_csv("./data_papers2.csv")
paper_data.columns=["country"]+list(paper_data.columns[1:])
paper_data.sort_values(by="country",inplace=True)
paper_data.head()


# In[10]:


results.head()


# In[11]:


choice=results
choice.sort_values(by="country",inplace=True)
#choice.drop(9,inplace=True)#China is duplicated


choice["paperpopulation"]=paper_data["children"].values
choice["papercases"]=paper_data["NO2cases"].values
choice["paperincidence"]=paper_data["NO2incidence"].values


# In[12]:


plt.figure(figsize=(12,10))
import seaborn as sns
x=choice["1-18 population"]
y=choice["paperpopulation"]
sns.scatterplot(data=choice,x="1-18 population",y="paperpopulation",hue="country")
plt.xlabel("Our module",fontsize=20)
plt.ylabel("Achakulwisut et al",fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("1-18y population",fontsize=32)
plt.plot([0,max(max(x),max(y))],[0,max(max(x),max(y))],"k-")
#plt.savefig("/home/gaia/Desktop/asthma/population_comparison.png")
plt.show()


# In[13]:


plt.figure(figsize=(12,10))
import seaborn as sns
x=choice["NO2cases"]
y=choice["papercases"]
sns.scatterplot(data=choice,x="NO2cases",y="papercases",hue="country")
plt.xlabel("Our module",fontsize=20)
plt.ylabel("Achakulwisut et al",fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Asthma NO2 cases",fontsize=32)
plt.plot([0,max(max(x),max(y))],[0,max(max(x),max(y))],"k-")
plt.savefig("/home/gaia/Desktop/asthma/EMEP_cases_comparison_threshold5.png")
plt.show()


# In[17]:


choice


# In[14]:


plt.figure(figsize=(12,10))
import seaborn as sns
x=choice["NO2incidence"]
y=choice["paperincidence"]
sns.scatterplot(data=choice,x="NO2incidence",y="paperincidence",hue="country",s=80)
plt.xlabel("Our module",fontsize=20)
plt.ylabel("Achakulwisut et al",fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Asthma NO2 incidence",fontsize=32)
plt.plot([0,max(max(x),max(y))],[0,max(max(x),max(y))],"k-")
plt.savefig("/home/gaia/Desktop/asthma/EMEP_inc_comparison_threshold5.png")
plt.show()


# In[19]:


plt.figure(figsize=(12,10))
import seaborn as sns
results["year"]=[int(x) for x in results["year"]]
data=results.loc[results.scenario!="MFR",:]
sns.barplot(data=data,x="country",y="NO2incidence",hue="year")
plt.xlabel("Country",fontsize=20)
plt.ylabel("NO2 asthma incidence",fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("CLE - Asthma trends",fontsize=32)
plt.savefig("/home/gaia/Desktop/asthma/CE.png")
plt.show()


# In[20]:


plt.figure(figsize=(12,10))
import seaborn as sns
results["year"]=[int(x) for x in results["year"]]
data=results.loc[results.scenario!="CLE",:]
sns.barplot(data=data,x="country",y="NO2incidence",hue="year")
plt.xlabel("Country",fontsize=20)
plt.ylabel("NO2 asthma incidence",fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("MFR - Asthma trends",fontsize=32)
plt.savefig("/home/gaia/Desktop/asthma/MFR.png")
plt.show()


# In[4]:


# countries=[x for x in countries_masks if (x not in countries_un)&(x not in un_translate.keys()) ]
# countries
# with open("./data/non_included_countries.txt", "w") as file:
#     file.write(str(countries))

with open("./data/non_included_countries.txt", "r") as file:
    non_included_countries = eval(file.readline())


# In[6]:


scenario_dir="./data/population_projections/ssp1/"
scenarios=os.listdir(scenario_dir)
store_results="/media/gaia/Seagate Expansion Drive/Liliana/fon/asthma/population_projections/"

for scenario in scenarios:
    ds=gdal.Open(os.path.join(scenario_dir,scenario))
    ds = gdal.Translate(os.path.join(store_results,"corrected_"+scenario), ds, 
                        projWin = [-180, 90, 180, -90],format="GTiff",outputType=gdalconst.GDT_Int32)
    ds=None


# In[12]:


nox_dir="/media/gaia/Seagate Expansion Drive/Liliana/fon/Downscaling"
scenarios=os.listdir(scenario_dir)
store_results="/media/gaia/Seagate Expansion Drive/Liliana/fon/asthma/population_projections/"

for scenario in scenarios:
    ds=gdal.Open(os.path.join(scenario_dir,scenario))
    ds = gdal.Translate(os.path.join(store_results,"corrected_"+scenario), ds, 
                        projWin = [-180, 90, 180, -90],format="GTiff",outputType=gdalconst.GDT_Int32)
    ds=None


# In[26]:


nox_dir="/media/gaia/Seagate Expansion Drive/Liliana/fon/Downscaling"
scenarios=os.listdir(nox_dir)[:10]
for scenario in scenarios:
    ds=gdal.Open(os.path.join(nox_dir,scenario))
    ds = gdal.Translate(os.path.join(nox_dir,"extended","extended"+scenario), ds, 
                        projWin = [-180, 90, 180, -90])
    ds=None


# In[20]:


1-np.exp(-10*beta)


# In[21]:


beta

