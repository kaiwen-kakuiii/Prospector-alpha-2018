
def load_data(name,data_list,**extras): 
    import time, sys, os
    import h5py
    import numpy as np
    import scipy
    from matplotlib.pyplot import *
    import fsps
    import sedpy
    import prospect
    import emcee
    import astropy
    import math
        
    from astropy.io.votable import parse
    votable = parse(name)
    table = votable.get_first_table()
    data=table.array
    raw_data={}
    leng = len(data)
    
    # construct a list of all frequency to sort out the order
    raw_data['filtername']=[]
    raw_data['flux']=[]
    raw_data['unc']=[]
    freq = []
    name_tem = []
    maxi=[]
    for table in votable.iter_tables():

        for i in range(0,leng):
            name = data[i][7].lower()
            f = data[i][4]
            length = len(name)
            Filter = ''
            for j in range(0,length):
                if name[j]==":":
                    Filter+="_"
                elif name[j]=="/":
                    Filter+="_"
                elif name[j]=="'":
                    pass
                else:
                    Filter+=name[j]
            #print(data[i][4])
            
            # We combine some filters and set the frequence conrrespondingly.
            if Filter == 'pan-starrs_ps1_r':
                Filter = 'sdss_r'
                f = 4.799e5
            elif Filter == 'pan-starrs_ps1_g':
                Filter = 'sdss_g'
                f = 6.2198e5
            elif Filter == 'pan-starrs_ps1_i':
                Filter = 'sdss_i'
                f = 3.9266e5
            elif Filter == 'vista_j':
                Filter = 'ukidss_j'
                f = 2.4016e5
            elif Filter == 'vista_h':
                Filter = 'ukidss_h'
                f = 1.8378e5
            else:
                pass
            name_tem.append(Filter)
            freq.append(f)
    
        for i in range(0,leng):
            maxi.append(freq[i])
            
        for k in range(0,leng):
            max_value = max(maxi)
            max_index = maxi.index(max_value)
            # then we can store the data to raw_data for later use
            # change the filter name 
            raw_data['flux'].append(data[max_index][5])
            raw_data['filtername'].append(name_tem[max_index])
            
            # Create error to some bins 
            if type(data[max_index][6])!= np.float32:
                raw_data['unc'].append((data[max_index][5])/5)
                #print('good')
            else:
                raw_data['unc'].append(data[max_index][6])
                
            maxi[max_index]=0
        pass
    print(len(raw_data['filtername']))

    # Now we need to combine/average the data for same filter and propogate the error
    # we can also pre-process our data here 
    fit_data={}
    fit_data['filtername']=[]
    fit_data['flux']=[]
    fit_data['unc']=[]
    i=0
    while i<leng:
        flux = 0
        unc = 0
        unc2=0
        ### select the data we need for the fit        
        if data_list.count(raw_data['filtername'][i])==0:
            i+=1
            continue
        ###
        
        time = raw_data['filtername'].count(raw_data['filtername'][i])
    
        if raw_data['filtername'][i][-1]=="'":  
            time += raw_data['filtername'].count(raw_data['filtername'][i][:-1])
            fit_data['filtername'].append(raw_data['filtername'][i][:-1])
        else:
            time += raw_data['filtername'].count(raw_data['filtername'][i]+"'")
            fit_data['filtername'].append(raw_data['filtername'][i])
            
        for j in range(i,i+time):
            flux += (raw_data['flux'][j])/time
            
        fit_data['flux'].append(flux)
    
        for k in range(i,i+time):
            #unc1 = max(raw_data['flux'][i:i+time])-min(raw_data['flux'][i:i+time])
            #unc2 += raw_data['unc'][k]
            #unc = max(unc1,unc2)
            unc += raw_data['unc'][k]/time

        unc = max(unc,flux/5)
        fit_data['unc'].append(unc)
        i+=time
    
    #print((fit_data['filtername']))

    return fit_data

#################################################################################################

def build_obs(fit_data, snr=10,  **extras):
    import numpy as np
    import scipy
    import fsps
    import astropy
    import math

    from prospect.utils.obsutils import fix_obs
    import sedpy

    # The obs dictionary, empty for now
    obs = {}

    # These are the names of the relevant filters,  
    # in the same order as the photometric data (see below)
    # And here we instantiate the `Filter()` objects using methods in `sedpy`,
    # and put the resultinf list of Filter objects in the "filters" key of the `obs` dictionary
    obs["filters"] = sedpy.observate.load_filters(fit_data['filtername'])
    # Now we store the measured fluxes for a single object, **in the same order as "filters"**
    # In this example we use a row of absolute AB magnitudes from Johnson et al. 2013 (NGC4163)
    # We then turn them into apparent magnitudes based on the supplied `ldist` meta-parameter.
    # You could also, e.g. read from a catalog.
    # The units of the fluxes need to be maggies (Jy/3631) so we will do the conversion here too.
    
    obs["maggies"] = [x / 3631 for x in fit_data['flux']]
    obs['maggies']=np.array(obs['maggies'])
    
    # And now we store the uncertainties (again in units of maggies)
    # In this example we are going to fudge the uncertainties based on the supplied `snr` meta-parameter.
    obs["maggies_unc"] = [x / 3631 for x in fit_data["unc"]]
    obs['maggies_unc']=np.array(obs['maggies_unc'])

    # Now we need a mask, which says which flux values to consider in the likelihood.
    # IMPORTANT: the mask is *True* for values that you *want* to fit, 
    # and *False* for values you want to ignore.  Here we ignore the spitzer bands.
    obs["phot_mask"] = np.array([True for f in obs["filters"]])
    
    # This is an array of effective wavelengths for each of the filters.  
    # It is not necessary, but it can be useful for plotting so we store it here as a convenience
    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])

    # We do not have a spectrum, so we set some required elements of the obs dictionary to None.
    # (this would be a vector of vacuum wavelengths in angstroms)
    obs["wavelength"] = None
    # (this would be the spectrum in units of maggies)
    obs["spectrum"] = None
    # (spectral uncertainties are given here)
    obs['unc'] = None
    # (again, to ignore a particular wavelength set the value of the 
    #  corresponding elemnt of the mask to *False*)
    obs['mask'] = None

    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)
    
    # Due to some unknown reason, fix_obs will change true to false in phot_mask.We do it again after the function.
    obs["phot_mask"] = np.array([True for f in obs["filters"]])
    
    return obs

###############################################################################################

def build_model(object_redshift=None,  fixed_metallicity=None, add_duste=False, 
                **extras):
    """Build a prospect.models.SedModel object
    
    :param object_redshift: (optional, default: None)
        If given, produce spectra and observed frame photometry appropriate 
        for this redshift. Otherwise, the redshift will be zero.
        
    :param ldist: (optional, default: 10)
        The luminosity distance (in Mpc) for the model.  Spectra and observed 
        frame (apparent) photometry will be appropriate for this luminosity distance.
        
    :param fixed_metallicity: (optional, default: None)
        If given, fix the model metallicity (:math:`log(Z/Z_sun)`) to the given value.
    
    :param add_duste: (optional, default: False)
        If `True`, add dust emission and associated (fixed) parameters to the model.
        
    :returns model:
        An instance of prospect.models.SedModel
    """
    from prospect.models.sedmodel import SedModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors
    from prospect.models import transforms
    from prospect.models.templates import adjust_continuity_agebins
    from astropy.cosmology import WMAP9 as cosmos
    import prospect
    # Get (a copy of) one of the prepackaged model set dictionaries.
    # Get the 2018 prospector-alpha model manually
    model_params = (TemplateLibrary["continuity_sfh"])
    model_params.update(TemplateLibrary["dust_emission"])
    model_params.update(TemplateLibrary["nebular"])
    model_params.update(TemplateLibrary["agn"])

    # Set the dust and agn emission free
    model_params["fagn"]["isfree"] = True
    model_params["agn_tau"]["isfree"] = True

    # Complexify the dust attenuation
    model_params["dust_type"] = {"N": 1, "isfree": False, "init": 4, "units": "FSPS index"}
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)
    model_params["dust1"]      = {"N": 1, "isfree": False, 'depends_on': transforms.dustratio_to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars"}

    model_params["dust_ratio"] = {"N": 1, "isfree": True, 
                             "init": 1.0, "units": "ratio of birth-cloud to diffuse dust",
                             "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    model_params["dust_index"] = {"N": 1, "isfree": True,
                             "init": 0.0, "units": "power-law multiplication of Calzetti",
                             "prior": priors.TopHat(mini=-2.0, maxi=0.5)}
    # in Gyr
    tuniv = cosmos.age(object_redshift).value
    model_params = adjust_continuity_agebins(model_params, tuniv)
    
    model_params["duste_qpah"]["isfree"] = False
    model_params["duste_umin"]["isfree"] = False
    model_params["duste_gamma"]["isfree"] = False
    model_params["duste_qpah"]["init"] = 2.0
    model_params["duste_umin"]["init"] = 1.0
    model_params["duste_gamma"]["init"] = 0.01
    model_params["duste_qpah"]["prior"] = priors.TopHat(mini=0.0, maxi=7.0)
    model_params["duste_umin"]["prior"] = priors.TopHat(mini=0.1, maxi=25.0)
    model_params["duste_gamma"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
    model_params["duste_qpah"]["disp_floor"] = 3.0
    model_params["duste_umin"]["disp_floor"] = 4.5
    model_params["duste_gamma"]["disp_floor"] = 0.15
    model_params["duste_qpah"]["init_disp"] = 3.0
    model_params["duste_umin"]["init_disp"] = 5.0
    model_params["duste_gamma"]["init_disp"] = 0.2
    model_params['gas_logz']["isfree"] = True
    model_params['gas_logz']["init"] = 0.0
    model_params['gas_logz']["prior"] = priors.TopHat(mini=-2.0, maxi=0.5) 
    model_params['gas_logu']["isfree"] = False
    model_params['gas_logu']["init"] = -1.0
    model_params['gas_logu']["prior"] = priors.TopHat(mini=-4.0, maxi=-1.0)
    
   # Now add the lumdist parameter by hand as another entry in the dictionary.
   # This will control the distance since we are setting the redshift to zero.  
   # In `build_obs` above we used a distance of 10Mpc to convert from absolute to apparent magnitudes, 
   # so we use that here too, since the `maggies` are appropriate for that distance.
    
    # Let's make some changes to values appropriate for our objects and data
    model_params['dust_type']['init']=4
    model_params['fagn']['init']=0.5
    model_params['dust2']['init']=0.3
    model_params['sfh']['init']=0
  
    model_params["dust_index"]["prior"] = priors.TopHat(mini=-2.2, maxi=0.4)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2.0, maxi=0.2)
    model_params["dust2"]["prior"] = priors.TopHat(mini=1e-6, maxi=3.0)
    model_params["agn_tau"]["prior"] = priors.LogUniform(mini=5, maxi=1.5e2)
    # If we are going to be using emcee, it is useful to provide a 
    # minimum scale for the cloud of walkers (the default is 0.1)
    model_params["agn_tau"]["disp_floor"] = 1e-2
    model_params["dust2"]["disp_floor"] = 1e-2
    model_params["logzsol"]["disp_floor"] = 1e-3
    model_params['fagn']['disp_floor'] = 1e-3
    model_params['dust_index']['disp_floor']=1e-3
    # Change the model parameter specifications based on some keyword arguments

    if object_redshift is not None:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    #print((model_params))
    
    # Change fit orders 
    tparams = {}
    parnames = [m for m in model_params]
    fit_order = ['logmass','dust2', 'logzsol','fagn','dust_index', 'dust_ratio']
    for param in fit_order:
        tparams[param]=model_params[param]
    for param in model_params:
        if param not in fit_order:
            tparams[param] = model_params[param]
    model_params = tparams 
    
    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)
    return model

###########################################################################################
def build_model_neb(object_redshift=None,  fixed_metallicity=None, add_duste=False, 
                **extras):

    from prospect.models.sedmodel import SedModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors
    from prospect.models import transforms
    from prospect.models.templates import adjust_continuity_agebins
    from astropy.cosmology import WMAP9 as cosmos
    import prospect
    # Get (a copy of) one of the prepackaged model set dictionaries.
    # Get the 2018 prospector-alpha model manually
    model_params = (TemplateLibrary["continuity_sfh"])
    model_params.update(TemplateLibrary["dust_emission"])
    model_params.update(TemplateLibrary["nebular"])
    model_params.update(TemplateLibrary["agn"])

    # Set the dust and agn emission free
    model_params["fagn"]["isfree"] = True
    model_params["agn_tau"]["isfree"] = True

    # Complexify the dust attenuation
    model_params["dust_type"] = {"N": 1, "isfree": False, "init": 4, "units": "FSPS index"}
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)
    model_params["dust1"]      = {"N": 1, "isfree": False, 'depends_on': transforms.dustratio_to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars"}

    model_params["dust_ratio"] = {"N": 1, "isfree": True, 
                             "init": 1.0, "units": "ratio of birth-cloud to diffuse dust",
                             "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    model_params["dust_index"] = {"N": 1, "isfree": True,
                             "init": 0.0, "units": "power-law multiplication of Calzetti",
                             "prior": priors.TopHat(mini=-2.0, maxi=0.5)}
    # in Gyr
    tuniv = cosmos.age(object_redshift).value
    model_params = adjust_continuity_agebins(model_params, tuniv)
    
    model_params["duste_qpah"]["isfree"] = False
    model_params["duste_umin"]["isfree"] = False
    model_params["duste_gamma"]["isfree"] = False
    model_params["duste_qpah"]["init"] = 2.0
    model_params["duste_umin"]["init"] = 1.0
    model_params["duste_gamma"]["init"] = 0.01
    model_params["duste_qpah"]["prior"] = priors.TopHat(mini=0.0, maxi=7.0)
    model_params["duste_umin"]["prior"] = priors.TopHat(mini=0.1, maxi=25.0)
    model_params["duste_gamma"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
    model_params["duste_qpah"]["disp_floor"] = 3.0
    model_params["duste_umin"]["disp_floor"] = 4.5
    model_params["duste_gamma"]["disp_floor"] = 0.15
    model_params["duste_qpah"]["init_disp"] = 3.0
    model_params["duste_umin"]["init_disp"] = 5.0
    model_params["duste_gamma"]["init_disp"] = 0.2
    model_params['gas_logz']["isfree"] = True
    model_params['gas_logz']["init"] = 0.0
    model_params['gas_logz']["prior"] = priors.TopHat(mini=-2.0, maxi=0.5) 
    model_params['gas_logu']["isfree"] = False
    model_params['gas_logu']["init"] = -1.0
    model_params['gas_logu']["prior"] = priors.TopHat(mini=-4.0, maxi=-1.0)
   # Now add the lumdist parameter by hand as another entry in the dictionary.
   # This will control the distance since we are setting the redshift to zero.  
   # In `build_obs` above we used a distance of 10Mpc to convert from absolute to apparent magnitudes, 
   # so we use that here too, since the `maggies` are appropriate for that distance.
    #model_params["lumdist"] = {"N": 1, "isfree": False, "init": ldist, "units":"Mpc"}
    
    # Let's make some changes to values appropriate for our objects and data

    model_params['dust_type']['init']=0
    model_params['fagn']['init']=0.5
    model_params['dust2']['init']=0.3
    model_params['sfh']['init']=0

    model_params["dust_index"]["prior"] = priors.TopHat(mini=-2.2, maxi=0.4)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2.0, maxi=0.2)
    model_params["dust2"]["prior"] = priors.TopHat(mini=1e-6, maxi=3.0)
    model_params["agn_tau"]["prior"] = priors.LogUniform(mini=5, maxi=1.5e2)
    # If we are going to be using emcee, it is useful to provide a 
    # minimum scale for the cloud of walkers (the default is 0.1)
    model_params["agn_tau"]["disp_floor"] = 0.1
    model_params["dust2"]["disp_floor"] = 1e-2
    model_params["logzsol"]["disp_floor"] = 1e-3
    model_params['fagn']['disp_floor'] = 1e-3
    model_params['dust_index']['disp_floor']=1e-3
    # Change the model parameter specifications based on some keyword arguments

    if object_redshift is not None:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    #print((model_params))
    
    # Change fit orders 
    tparams = {}
    parnames = [m for m in model_params]
    fit_order = ['logmass','dust2', 'logzsol','fagn','dust_index', 'dust_ratio']
    for param in fit_order:
        tparams[param]=model_params[param]
    for param in model_params:
        if param not in fit_order:
            tparams[param] = model_params[param]
    model_params = tparams 
    
    ########
    model_params['add_dust_emission']['init']= False
    model_params['add_neb_emission']['init']= False
    model_params['nebemlineinspec']['init']= False
    ########
    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)
    return model

###########################################################################################
def build_model_dust(object_redshift=None,fixed_metallicity=None, add_duste=False, 
                **extras):

    from prospect.models.sedmodel import SedModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors
    from prospect.models import transforms
    from prospect.models.templates import adjust_continuity_agebins
    from astropy.cosmology import WMAP9 as cosmos
    import prospect
    # Get (a copy of) one of the prepackaged model set dictionaries.
    # Get the 2018 prospector-alpha model manually
    model_params = (TemplateLibrary["continuity_sfh"])
    model_params.update(TemplateLibrary["dust_emission"])
    model_params.update(TemplateLibrary["nebular"])
    model_params.update(TemplateLibrary["agn"])

    # Set the dust and agn emission free
    model_params["fagn"]["isfree"] = True
    model_params["agn_tau"]["isfree"] = True

    # Complexify the dust attenuation
    model_params["dust_type"] = {"N": 1, "isfree": False, "init": 4, "units": "FSPS index"}
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)
    model_params["dust1"]      = {"N": 1, "isfree": False, 'depends_on': transforms.dustratio_to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars"}

    model_params["dust_ratio"] = {"N": 1, "isfree": True, 
                             "init": 1.0, "units": "ratio of birth-cloud to diffuse dust",
                             "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    model_params["dust_index"] = {"N": 1, "isfree": True,
                             "init": 0.0, "units": "power-law multiplication of Calzetti",
                             "prior": priors.TopHat(mini=-2.0, maxi=0.5)}
    # in Gyr
    tuniv = cosmos.age(object_redshift).value
    model_params = adjust_continuity_agebins(model_params, tuniv)
    
    model_params["duste_qpah"]["isfree"] = False
    model_params["duste_umin"]["isfree"] = False
    model_params["duste_gamma"]["isfree"] = False
    model_params["duste_qpah"]["init"] = 2.0
    model_params["duste_umin"]["init"] = 1.0
    model_params["duste_gamma"]["init"] = 0.01
    model_params["duste_qpah"]["prior"] = priors.TopHat(mini=0.0, maxi=7.0)
    model_params["duste_umin"]["prior"] = priors.TopHat(mini=0.1, maxi=25.0)
    model_params["duste_gamma"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
    model_params["duste_qpah"]["disp_floor"] = 3.0
    model_params["duste_umin"]["disp_floor"] = 4.5
    model_params["duste_gamma"]["disp_floor"] = 0.15
    model_params["duste_qpah"]["init_disp"] = 3.0
    model_params["duste_umin"]["init_disp"] = 5.0
    model_params["duste_gamma"]["init_disp"] = 0.2
    model_params['gas_logz']["isfree"] = True
    model_params['gas_logz']["init"] = 0.0
    model_params['gas_logz']["prior"] = priors.TopHat(mini=-2.0, maxi=0.5) 
    model_params['gas_logu']["isfree"] = False
    model_params['gas_logu']["init"] = -1.0
    model_params['gas_logu']["prior"] = priors.TopHat(mini=-4.0, maxi=-1.0)
   # Now add the lumdist parameter by hand as another entry in the dictionary.
   # This will control the distance since we are setting the redshift to zero.  
   # In `build_obs` above we used a distance of 10pc to convert from absolute to apparent magnitudes, 
   # so we use that here too, since the `maggies` are appropriate for that distance.
    
    # Let's make some changes to values appropriate for our objects and data
    
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2.0, maxi=0.2)
    model_params["dust_index"]["prior"] = priors.TopHat(mini=-2.2, maxi=0.4)
    model_params["agn_tau"]["prior"] = priors.LogUniform(mini=5, maxi=1.5e2)
    model_params["dust2"]["prior"] = priors.TopHat(mini=1e-6, maxi=3.0)

    model_params['dust_type']['init']=0
    model_params['fagn']['init']=0.5
    model_params['dust2']['init']=0.1
    # If we are going to be using emcee, it is useful to provide a 
    # minimum scale for the cloud of walkers (the default is 0.1)
    model_params["agn_tau"]["disp_floor"] = 1
    model_params["dust2"]["disp_floor"] = 1e-2
    model_params["logzsol"]["disp_floor"] = 1e-3
    model_params['fagn']['disp_floor'] = 1e-3
    model_params['dust_index']['disp_floor']=1e-3
    
    # Change the model parameter specifications based on some keyword arguments
    if object_redshift is not None:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift
    
     # Change fit orders 
    tparams = {}
    parnames = [m for m in model_params]
    fit_order = ['logmass','dust_index', 'dust2', 'logzsol','fagn' ,'dust_ratio']
    for param in fit_order:
        tparams[param]=model_params[param]
    for param in model_params:
        if param not in fit_order:
            tparams[param] = model_params[param]
    model_params = tparams
    
    ########
    model_params['add_dust_emission']['init']= False
    ########
    
    model_params['sfh']['init']=0
    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)
    #print(model_params['agebins'])
    return model
###########################################################################################
def build_model_att(object_redshift=None,fixed_metallicity=None, add_duste=False, 
                **extras):

    from prospect.models.sedmodel import SedModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors
    from prospect.models import transforms
    from prospect.models.templates import adjust_continuity_agebins
    from astropy.cosmology import WMAP9 as cosmos
    import prospect
    # Get (a copy of) one of the prepackaged model set dictionaries.
    # Get the 2018 prospector-alpha model manually
    model_params = (TemplateLibrary["continuity_sfh"])
    model_params.update(TemplateLibrary["dust_emission"])
    model_params.update(TemplateLibrary["nebular"])
    model_params.update(TemplateLibrary["agn"])

    # Set the dust and agn emission free
    model_params["fagn"]["isfree"] = True
    model_params["agn_tau"]["isfree"] = True

    # Complexify the dust attenuation
    model_params["dust_type"] = {"N": 1, "isfree": False, "init": 4, "units": "FSPS index"}
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)
    model_params["dust1"]      = {"N": 1, "isfree": False, 'depends_on': transforms.dustratio_to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars"}

    model_params["dust_ratio"] = {"N": 1, "isfree": True, 
                             "init": 1.0, "units": "ratio of birth-cloud to diffuse dust",
                             "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    model_params["dust_index"] = {"N": 1, "isfree": True,
                             "init": 0.0, "units": "power-law multiplication of Calzetti",
                             "prior": priors.TopHat(mini=-2.0, maxi=0.5)}
    # in Gyr
    tuniv = cosmos.age(object_redshift).value
    model_params = adjust_continuity_agebins(model_params, tuniv)
    
    model_params["duste_qpah"]["isfree"] = False
    model_params["duste_umin"]["isfree"] = False
    model_params["duste_gamma"]["isfree"] = False
    model_params["duste_qpah"]["init"] = 2.0
    model_params["duste_umin"]["init"] = 1.0
    model_params["duste_gamma"]["init"] = 0.01
    model_params["duste_qpah"]["prior"] = priors.TopHat(mini=0.0, maxi=7.0)
    model_params["duste_umin"]["prior"] = priors.TopHat(mini=0.1, maxi=25.0)
    model_params["duste_gamma"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
    model_params["duste_qpah"]["disp_floor"] = 3.0
    model_params["duste_umin"]["disp_floor"] = 4.5
    model_params["duste_gamma"]["disp_floor"] = 0.15
    model_params["duste_qpah"]["init_disp"] = 3.0
    model_params["duste_umin"]["init_disp"] = 5.0
    model_params["duste_gamma"]["init_disp"] = 0.2
    model_params['gas_logz']["isfree"] = True
    model_params['gas_logz']["init"] = 0.0
    model_params['gas_logz']["prior"] = priors.TopHat(mini=-2.0, maxi=0.5) 
    model_params['gas_logu']["isfree"] = False
    model_params['gas_logu']["init"] = -1.0
    model_params['gas_logu']["prior"] = priors.TopHat(mini=-4.0, maxi=-1.0)
   # Now add the lumdist parameter by hand as another entry in the dictionary.
   # This will control the distance since we are setting the redshift to zero.  
   # In `build_obs` above we used a distance of 10pc to convert from absolute to apparent magnitudes, 
   # so we use that here too, since the `maggies` are appropriate for that distance.
    
    # Let's make some changes to values appropriate for our objects and data
    
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2.0, maxi=0.2)
    model_params["dust_index"]["prior"] = priors.TopHat(mini=-2.2, maxi=0.4)
    model_params["agn_tau"]["prior"] = priors.LogUniform(mini=5, maxi=1.5e2)
    model_params["dust2"]["prior"] = priors.TopHat(mini=1e-6, maxi=3.0)

    model_params['dust_type']['init']=0
    model_params['fagn']['init']=0.5
    model_params['dust2']['init']=0.1
    # If we are going to be using emcee, it is useful to provide a 
    # minimum scale for the cloud of walkers (the default is 0.1)
    model_params["agn_tau"]["disp_floor"] = 1
    model_params["dust2"]["disp_floor"] = 1e-2
    model_params["logzsol"]["disp_floor"] = 1e-3
    model_params['fagn']['disp_floor'] = 1e-3
    model_params['dust_index']['disp_floor']=1e-3
    

    # Change the model parameter specifications based on some keyword arguments
    if object_redshift is not None:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift
    
     # Change fit orders 
    tparams = {}
    parnames = [m for m in model_params]
    fit_order = ['logmass','dust_index', 'dust2', 'logzsol','fagn','dust_ratio']
    for param in fit_order:
        tparams[param]=model_params[param]
    for param in model_params:
        if param not in fit_order:
            tparams[param] = model_params[param]
    model_params = tparams 
    ########
    model_params['add_dust_emission']['init']= False
    model_params['add_neb_emission']['init']= False
    ########
    model_params['sfh']['init']=0
    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)
    #print(model_params['agebins'])
    return model
###########################################################################################
def build_model_un(object_redshift=None,fixed_metallicity=None, add_duste=False, 
                **extras):

    from prospect.models.sedmodel import SedModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors
    from prospect.models import transforms
    from prospect.models.templates import adjust_continuity_agebins
    from astropy.cosmology import WMAP9 as cosmos
    import prospect
    # Get (a copy of) one of the prepackaged model set dictionaries.
    # Get the 2018 prospector-alpha model manually
    model_params = (TemplateLibrary["continuity_sfh"])
    model_params.update(TemplateLibrary["dust_emission"])
    model_params.update(TemplateLibrary["nebular"])
    model_params.update(TemplateLibrary["agn"])

    # Set the dust and agn emission free
    model_params["fagn"]["isfree"] = True
    model_params["agn_tau"]["isfree"] = True

    # Complexify the dust attenuation
    model_params["dust_type"] = {"N": 1, "isfree": False, "init": 4, "units": "FSPS index"}
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)
    model_params["dust1"]      = {"N": 1, "isfree": False, 'depends_on': transforms.dustratio_to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars"}

    model_params["dust_ratio"] = {"N": 1, "isfree": True, 
                             "init": 1.0, "units": "ratio of birth-cloud to diffuse dust",
                             "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    model_params["dust_index"] = {"N": 1, "isfree": True,
                             "init": 0.0, "units": "power-law multiplication of Calzetti",
                             "prior": priors.TopHat(mini=-2.0, maxi=0.5)}
    # in Gyr
    tuniv = cosmos.age(object_redshift).value
    model_params = adjust_continuity_agebins(model_params, tuniv)
    
    model_params["duste_qpah"]["isfree"] = False
    model_params["duste_umin"]["isfree"] = False
    model_params["duste_gamma"]["isfree"] = False
    model_params["duste_qpah"]["init"] = 2.0
    model_params["duste_umin"]["init"] = 1.0
    model_params["duste_gamma"]["init"] = 0.01
    model_params["duste_qpah"]["prior"] = priors.TopHat(mini=0.0, maxi=7.0)
    model_params["duste_umin"]["prior"] = priors.TopHat(mini=0.1, maxi=25.0)
    model_params["duste_gamma"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
    model_params["duste_qpah"]["disp_floor"] = 3.0
    model_params["duste_umin"]["disp_floor"] = 4.5
    model_params["duste_gamma"]["disp_floor"] = 0.15
    model_params["duste_qpah"]["init_disp"] = 3.0
    model_params["duste_umin"]["init_disp"] = 5.0
    model_params["duste_gamma"]["init_disp"] = 0.2
    model_params['gas_logz']["isfree"] = True
    model_params['gas_logz']["init"] = 0.0
    model_params['gas_logz']["prior"] = priors.TopHat(mini=-2.0, maxi=0.5) 
    model_params['gas_logu']["isfree"] = False
    model_params['gas_logu']["init"] = -1.0
    model_params['gas_logu']["prior"] = priors.TopHat(mini=-4.0, maxi=-1.0)
   # Now add the lumdist parameter by hand as another entry in the dictionary.
   # This will control the distance since we are setting the redshift to zero.  
   # In `build_obs` above we used a distance of 10pc to convert from absolute to apparent magnitudes, 
   # so we use that here too, since the `maggies` are appropriate for that distance.
    
    # Let's make some changes to values appropriate for our objects and data
    
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2.0, maxi=0.2)
    model_params["dust_index"]["prior"] = priors.TopHat(mini=-2.2, maxi=0.4)
    model_params["agn_tau"]["prior"] = priors.LogUniform(mini=5, maxi=1.5e2)
    model_params["dust2"]["prior"] = priors.TopHat(mini=1e-6, maxi=3.0)

    model_params['dust_type']['init']=0
    model_params['fagn']['init']=0.5
    model_params['dust2']['init']=0.1
    # If we are going to be using emcee, it is useful to provide a 
    # minimum scale for the cloud of walkers (the default is 0.1)
    model_params["agn_tau"]["disp_floor"] = 1
    model_params["dust2"]["disp_floor"] = 1e-2
    model_params["logzsol"]["disp_floor"] = 1e-3
    model_params['fagn']['disp_floor'] = 1e-3
    model_params['dust_index']['disp_floor']=1e-3
    
    # Change the model parameter specifications based on some keyword arguments
    if object_redshift is not None:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift
    
     # Change fit orders 
    tparams = {}
    parnames = [m for m in model_params]
    fit_order = ['logmass','dust_index', 'dust2', 'logzsol','fagn' ,'dust_ratio']
    for param in fit_order:
        tparams[param]=model_params[param]
    for param in model_params:
        if param not in fit_order:
            tparams[param] = model_params[param]
    model_params = tparams 
    ########
    model_params['add_neb_emission']['init']= False
    ########
    model_params['sfh']['init']=0
    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)
    #print(model_params['agebins'])
    return model
###########################################################################################
def build_sps(zcontinuous=1, **extras):
    """
    :param zcontinuous: 
        A vlue of 1 insures that we use interpolation between SSPs to 
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    from prospect.sources import FastStepBasis
    sps = FastStepBasis(zcontinuous=zcontinuous)
    return sps

###########################################################################################

def viewer(file_name):
    import time, sys, os
    import h5py
    import numpy as np
    import scipy
    from matplotlib.pyplot import *
    import  matplotlib.pyplot as plt
    import fsps
    import sedpy
    import prospect
    import emcee
    import astropy
    import math
    from func import *
    import prospect.io.read_results as reader
    results_type = "emcee" # | "dynesty"
    # grab results (dictionary), the obs dictionary, and our corresponding models
    # When using parameter files set `dangerous=True`
    tem_name = file_name[:-8]+'{}.h5'
    #result, obs, _ = reader.results_from(tem_name.format(results_type), dangerous=False)
    result, obs, _ = reader.results_from(file_name, dangerous=False)

    #The following commented lines reconstruct the model and sps object, 
    # if a parameter file continaing the `build_*` methods was saved along with the results
    #model = reader.get_model(result)
    #sps = reader.get_sps(result)
    
    # let's look at what's stored in the `result` dictionary
    print(result.keys())

    if results_type == "emcee":
        chosen = np.random.choice(result["run_params"]["nwalkers"], size=10, replace=False)
        tracefig = reader.traceplot(result, figsize=(20,10), chains=chosen)
    else:
        tracefig = reader.traceplot(result, figsize=(20,10))

    # maximum a posteriori (of the locations visited by the MCMC sampler)
    imax = np.argmax(result['lnprobability'])
    if results_type == "emcee":
        i, j = np.unravel_index(imax, result['lnprobability'].shape)
        theta_max = result['chain'][i, j, :].copy()
        thin = 5
    else:
        theta_max = result["chain"][imax, :]
        thin = 1

    print('MAP value: {}'.format(theta_max))
    cornerfig = reader.subcorner(result, start=0, thin=thin, 
                             fig=subplots(15,15,figsize=(27,27))[0])

    # --- Draw final fitting curve ---
    
    run_params = result['run_params']
    print(run_params)
    model = build_model(**run_params)
    print(model)
    sps = build_sps(**run_params)
    
    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
    # photometric effective wavelengths
    wphot = obs["phot_wave"]
    # spectroscopic wavelengths
    if obs["wavelength"] is None:
        # *restframe* spectral wavelengths, since obs["wavelength"] is None
        wspec = sps.wavelengths
        wspec *= a #redshift them
    else:
        wspec = obs["wavelength"]

    # Get enssential data to calculate sfr
    z_fraction = theta_max[5:10]
    total_mass = theta_max[0]
    agebins = model.params['agebins']
    x = np.zeros(6)
    for i in range(6):
        if i ==5:
            x[i] = (agebins[i][0]+agebins[i][1])/2
        else:
            x[i]=(agebins[i][0]+agebins[i+1][0])/2
    
    sfr = prospect.models.transforms.zfrac_to_sfr(total_mass, z_fraction, agebins)

    # Calculate the 16% and 84% value from emcee results
    z_fraction1 = []
    z_fraction2 = []
    z_fraction3 = []
    z_fraction4 = []
    z_fraction5 = []
    for i in range(30):
        for j in range(3000):
            z_fraction1.append(result['chain'][i][j][5])
            z_fraction2.append(result['chain'][i][j][6])
            z_fraction3.append(result['chain'][i][j][7])
            z_fraction4.append(result['chain'][i][j][8])
            z_fraction5.append(result['chain'][i][j][9])

    zerr_1 = [np.percentile(z_fraction1,16), np.percentile(z_fraction1,84)]
    zerr_2 = [np.percentile(z_fraction2,16), np.percentile(z_fraction2,84)]
    zerr_3 = [np.percentile(z_fraction3,16), np.percentile(z_fraction3,84)]
    zerr_4 = [np.percentile(z_fraction4,16), np.percentile(z_fraction4,84)]
    zerr_5 = [np.percentile(z_fraction5,16), np.percentile(z_fraction5,84)] 
    # Build upper and lower z_fraction array
    z_max = np.zeros(5)
    z_min = np.zeros(5)
    for i in range(5):
        z_min[i]=eval('zerr_{}[0]'.format(i+1))
        z_max[i]=eval('zerr_{}[1]'.format(i+1))
    #print(z_min)
    #print(z_max)

    # Build new sfr
    sfr_max = prospect.models.transforms.zfrac_to_sfr(total_mass, z_max, agebins)
    sfr_min = prospect.models.transforms.zfrac_to_sfr(total_mass, z_min, agebins)

    # randomly chosen parameters from chain
    randint = np.random.randint    
    if results_type == "emcee":
        nwalkers, niter = run_params['nwalkers'], run_params['niter']
        # Draw 100 random plots from mcmc
        figure(figsize=(16,16))
        ax = plt.subplot(211)
        for i in range(100):
            # Get data from any random chain
            theta = result['chain'][randint(nwalkers), randint(niter)]
            mspec_t, mphot_t, mextra = model.mean_model(theta, obs, sps=sps)

            # unit conversion to erg/cm^2/s
            mspec = (mspec_t*1e-23)*3e18/wspec
            mphot = (mphot_t*1e-23)*3e18/wphot

            # plot them out 
            loglog(wspec, mspec,
                   lw=0.7, color='grey', alpha=0.3)        
    else:
        theta = result["chain"][randint(len(result["chain"]))]

    # now do it again for best fit model
    # sps = reader.get_sps(result)  # this works if using parameter files
    mspec_map_t, mphot_map_t, _ = model.mean_model(theta_max, obs, sps=sps)
    # units conversion to erg/cm^2/s
    mspec_map = (mspec_map_t*1e-23)*3e18/wspec
    mphot_map = (mphot_map_t*1e-23)*3e18/wphot
    ob = (obs['maggies']*1e-23)*3e18/wphot
    ob_unc = (obs['maggies_unc']*1e-23)*3e18/wphot

    # calculate reduced chi^2 
    chi = np.float128(0)
    for i in range(len(wphot)):
        var = ob_unc[i] 
        chi += np.float128((mphot_map[i] - ob[i])**2)/var**2

    # Make plot of best fit model
    loglog(wspec, mspec_map, label='Model spectrum (MAP)',lw=0.7, color='green', alpha=0.7)
    errorbar(wphot, mphot_map, label='Model photometry (MAP)',marker='s', markersize=10,
             alpha=0.8, ls='', lw=3, markerfacecolor='none', markeredgecolor='green',
             markeredgewidth=3)
    errorbar(wphot, ob, yerr=ob_unc, label='Observed photometry',
             ecolor='red', marker='o', markersize=10, ls='', lw=3, alpha=0.8, 
             markerfacecolor='none', markeredgecolor='red', markeredgewidth=3)
    text(0, 1,'Chi-squared/N-phot ='+ str(chi/len(wphot)),
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
    # Setup bound
    xmin, xmax = np.min(wphot)*0.6, np.max(wphot)/0.8
    temp = np.interp(np.linspace(xmin,xmax,10000), wphot, mphot_map)
    ymin, ymax = temp.min()*0.8, temp.max()*10
    xlabel('Wavelength [A]',fontsize=20)
    ylabel('Flux Density [erg/cm^2/s]',fontsize=20)
    xlim([xmin, xmax])
    ylim([ymin, ymax])
    legend(loc='lower right', fontsize=15)
    tight_layout()
    #  plt.show()
    
    # plot sfr on the same canvas
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    in_axes = inset_axes(ax, 
                           width="25%", # width = 30% of parent_bbox
                           height=2, # height : 1 inch
                           loc='upper center')
    in_axes.scatter(10**(x-9),sfr)
    #plt.fill_between(10**(x-9), sfr_min, sfr_max)
    xlabel('Age [Gyr]',fontsize=10)
    ylabel('SFR [M_sun/yr]',fontsize=10)
    #plt.xscale('log')
    plt.yscale('log')
    
    # plot residue as subplot
    figure(figsize=(16,8))
    plt.subplot(212,sharex=ax)
    import pylab
    residue = abs(mphot_map - ob)
    scatter(wphot,abs(mphot_map - ob), label= 'Residue',
            lw = 2, alpha = 0.8)
    plt.xscale('log')
    plt.yscale('log')
    xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
    temp = np.interp(np.linspace(xmin,xmax,10000), wphot, residue)
    ymin, ymax = temp.min()*0.1, temp.max()/0.8
    xlabel('Wavelength [A]',fontsize=20)
    ylabel('Flux Change [erg/cm^2/s]',fontsize=20)
    xlim([xmin, xmax])
    ylim([ymin, ymax])
    legend(loc='best', fontsize=15)
    tight_layout()
    return
#####################################################################################

def component(run_params,obs,mspec_map,theta_max):
    import fsps
    import sedpy
    import emcee
    import astropy
    import math
    from func import *
    import prospect
    sps = build_sps(**run_params)
  
    # dust-emission
    model_dust = build_model_dust(**run_params)
    mspec_dust_tt, mphot_dust_tt, mextra = model_dust.mean_model(theta_max, obs, sps=sps)
    mspec_dust_t = mspec_dust_tt*3631*1000
    mspec_map_dust = mspec_map - mspec_dust_t
    # neb-emission 
    model_neb = build_model_neb(**run_params)
    mspec_neb_tt, mphot_neb_tt, mextra = model_neb.mean_model(theta_max, obs, sps=sps)
    mspec_neb_t = mspec_neb_tt*3631*1000
    mspec_map_neb = mspec_dust_t - mspec_neb_t
    # agn-emission
    model = build_model(**run_params)
    theta_agn = theta_max.copy()
    theta_agn[6] = 0
    mspec_agn_tt, mphot_agn_tt, mextra = model.mean_model(theta_agn, obs, sps=sps)
    mspec_agn_t = mspec_agn_tt*3631*1000
    mspec_map_agn = mspec_map - mspec_agn_t
    # stellar-attenuate
    model_att = build_model_att(**run_params)
    theta_att = theta_max.copy()
    theta_att[6] = 0
    mspec_att_tt, mphot_att_tt, mextra = model_att.mean_model(theta_att, obs, sps=sps)
    mspec_att_t = mspec_att_tt*3631*1000
    mspec_map_att = mspec_att_t
    # stellar-unattenuate
    model_un = build_model_un(**run_params)
    theta_un = theta_max.copy()
    theta_un[2] = 0
    theta_un[6] = 0
    mspec_un_tt, mphot_un_tt, mextra = model_un.mean_model(theta_un, obs, sps=sps)
    mspec_un_t = mspec_un_tt*3631*1000
    mspec_map_un = mspec_un_t
    
    return mspec_map_neb,mspec_map_dust,mspec_map_agn,mspec_map_att,mspec_map_un
######################################################################################

def image(ra,dec,sur):
    from astropy import coordinates, units as u, wcs
    from astroquery.skyview import SkyView
    from astropy.io import fits
    from astropy.io.fits import getdata
    import aplpy
    from matplotlib.pyplot import *
    import numpy as np
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    
    center = coordinates.SkyCoord(str(ra)+' '+str(dec), frame='icrs',unit='deg')
    rad = 60 * u.arcsecond
    images = SkyView.get_images(position=center, survey=[sur],radius=rad)  
    x = np.zeros(1)
    y = np.zeros(1)
    r = np.zeros(1)
    x[0] = (ra)
    y[0] = (dec)
    r[0] = (0.00139)
    
    fig = aplpy.FITSFigure(images[0])
    fig.show_colorscale(cmap='gist_heat')
    fig.show_circles(x,y,r,color='g',alpha=1,lw=4)
    fig.recenter(ra,dec,radius=0.00416)
    fig.axis_labels.hide()
    fig.tick_labels.hide()
    fig.savefig(sur+'.png')
    #fig.set_theme('publication')

#######################################################################################

def fagn_mir(upp,med,low,run_params,theta_max,obs,wspec):
    import fsps
    import prospect
    import astropy
    import math
    from scipy.integrate import simps
    from func import *
    import numpy as np

    sps = build_sps(**run_params)

    model = build_model(**run_params)
    theta_agn = theta_max.copy()
    theta_agn[6] = 0
    mspec_agn_tt, mphot_agn_tt, mextra = model.mean_model(theta_agn, obs, sps=sps)
    mspec_agn_t_noagn = mspec_agn_tt*3631*1000
    #####################################
    model = build_model(**run_params)
    theta_agn = theta_max.copy()
    theta_agn[6] = med
    mspec_agn_tt, mphot_agn_tt, mextra = model.mean_model(theta_agn, obs, sps=sps)
    mspec_agn_t_med = mspec_agn_tt*3631*1000
    mspec_map_agn_med = mspec_agn_t_med - mspec_agn_t_noagn
    #####################################
    model = build_model(**run_params)
    theta_agn = theta_max.copy()
    theta_agn[6] = upp
    mspec_agn_tt, mphot_agn_tt, mextra = model.mean_model(theta_agn, obs, sps=sps)
    mspec_agn_t_upp = mspec_agn_tt*3631*1000
    mspec_map_agn_upp = mspec_agn_t_upp - mspec_agn_t_noagn
    ######################################
    model = build_model(**run_params)
    theta_agn = theta_max.copy()
    theta_agn[6] = low
    mspec_agn_tt, mphot_agn_tt, mextra = model.mean_model(theta_agn, obs, sps=sps)
    mspec_agn_t_low = mspec_agn_tt*3631*1000
    mspec_map_agn_low = mspec_agn_t_low - mspec_agn_t_noagn
 
    lamda = np.arange(4,20,1e-3)
    # median
    agn_ini_med = np.interp(lamda,wspec,mspec_map_agn_med)
    spec_ini_med = np.interp(lamda,wspec,mspec_agn_t_med)
    numbe_med = simps(agn_ini_med,lamda)
    denom = simps(spec_ini_med,lamda)
    mir_med = numbe_med/denom
    # upper
    agn_ini_upp = np.interp(lamda,wspec,mspec_map_agn_upp)
    spec_ini_upp = np.interp(lamda,wspec,mspec_agn_t_upp)
    numbe_upp = simps(agn_ini_upp,lamda)
    denom = simps(spec_ini_upp,lamda)
    mir_upp = numbe_upp/denom
    # lower
    agn_ini_low = np.interp(lamda,wspec,mspec_map_agn_low)
    spec_ini_low = np.interp(lamda,wspec,mspec_agn_t_low)
    numbe_low = simps(agn_ini_low,lamda)
    denom = simps(spec_ini_low,lamda)
    mir_low = numbe_low/denom
    
    return mir_med,mir_upp,mir_low
    
    
    
    
    
    
    
    
    
    
    
    
