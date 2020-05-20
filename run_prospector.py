def run_prospector(name_z):
    import time, sys, os
    import h5py	 
    import numpy as np
    import scipy
    from matplotlib.pyplot import *
    #%matplotlib inline
    import fsps
    import sedpy
    import prospect
    import emcee
    import astropy
    import math
    from prospect.fitting import fit_model
    from prospect.fitting import lnprobfn
    from func import *
    import matplotlib.pyplot as plt
    
    # --- Output files ---
    name = name_z[0]
    z = name_z[1]
    path_input = '/home/kk/work/data/red_magic/'+name
    path = '/home/kk/work/result/red_magic/'+name[:-4]
    '''try:
        os.mkdir(path)
    except:
        pass

    fe = open(path+'/para.txt','w')'''
    
    # --- Set up the essential parameter ---
    data_list = [ 'sdss_u', 'sdss_g', 'sdss_r','sdss_i', 'galex_fuv', 'galex_nuv', 'pan-starrs_ps1_i', 'pan-starrs_ps1_r',
                 'pan-starrs_ps1_g','pan-starrs_ps1_y', 'pan-starrs_ps1_z',
                     'sdss_z', 'ukidss_j','ukidss_h', 'ukidss_k', 'wise_w1', 'spitzer_irac_3.6', '_=3.6um', 
                     'spitzer_irac_4.5', '_=4.5um', 'wise_w2', 'spitzer_irac_5.8', 'spitzer_irac_8.0', 'wise_w3', 'wise_w4', 
                     'spitzer_mips_24','2mass_h','2mass_ks','2mass_j','gaia_gaia2_gbp','gaia_gaia2_g','gaia_gaia2_grp',
                 'gaia_g','ukidss_y','vista_j','vista_h','vista_ks']
    run_params = {}
    run_params["snr"] = 10.0
    run_params["object_redshift"] = z
    run_params["fixed_metallicity"] = None
    run_params["add_duste"] = True

    # --- Start build up the model ---
    fit_data=load_data(path_input, data_list)
    obs = build_obs(fit_data,**run_params)
    sps = build_sps(**run_params)
    model = build_model(**run_params)

    #fe.write(str(obs))
    #print(fit_data)
    # --- Draw initial data plot ---

    wphot = obs["phot_wave"]

    # establish bounds
    xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
    ymin, ymax = obs["maggies"].min()*0.8, obs["maggies"].max()/0.4
    figure(figsize=(16,8))
    
    # plot all the data
    plot(wphot, obs['maggies'],label='All observed photometry',marker='o',
         markersize=12, alpha=0.8, ls='', lw=3,color='slateblue')

    # overplot only the data we intend to fit
    mask = obs["phot_mask"]
    errorbar(wphot[mask], obs['maggies'][mask], yerr=obs['maggies_unc'][mask], 
         label='Photometry to fit',marker='o', markersize=8, alpha=0.8, ls='', lw=3,
         ecolor='tomato', markerfacecolor='none', markeredgecolor='tomato', markeredgewidth=3)

    # prettify
    xlabel('Wavelength [A]')
    ylabel('Flux Density [maggies]')
    xlim([xmin, xmax])
    ylim([ymin, ymax])
    xscale("log")
    yscale("log")
    legend(loc='best', fontsize=20)
    #tight_layout()
    #plt.savefig(path+'/input.png')

    # run minimization and emcee with error handling
    condition = False
    while condition is False:
        try:
            # --- start minimization ----
            run_params["dynesty"] = False
            run_params["emcee"] = False
            run_params["optimize"] = True
            run_params["min_method"] = 'lm'
            run_params["nmin"] = 5
            output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
            print("Done optmization in {}s".format(output["optimization"][1]))
          
            # save theta_best for later use
            (results, topt) = output["optimization"]
            ind_best = np.argmin([r.cost for r in results])
            theta_best = results[ind_best].x.copy()

            # --- start emcee ---
            run_params["optimize"] = False
            run_params["emcee"] = True
            run_params["dynesty"] = False
            run_params["nwalkers"] = 30
            run_params["niter"] = 8000
            run_params["nburn"] = [300, 800, 1600]
            output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
            print('done emcee in {0}s'.format(output["sampling"][1]))
            condition = True
            
        # sort out different errors and apply method to proceed
        except AssertionError as e:
            # error: sfr cannot be negative
            if str(e) == 'sfr cannot be negative.':
                obs = build_obs(fit_data,**run_params)
                sps = build_sps(**run_params)
                model = build_model(**run_params)
            # error: number of residules are less than variables
        except ValueError as e:
            if str(e) == "Method 'lm' doesn't work when the number of residuals is less than the number of variables.":
                print(name)
                return 
            # error: residule are not finite in the initial point
            elif str(e) == "Residuals are not finite in the initial point.":
                print(name)
                return
        
    # write final results to file
    from prospect.io import write_results as writer
    hfile = path + '_emcee.h5'
    print(hfile)
    writer.write_hdf5(hfile, run_params, model, obs,output["sampling"][0],output["optimization"][0],
                      tsample=output["sampling"][1],toptimize=output["optimization"][1])  
    #fe.write(str(obs))
    print('Finished')
    
    ##############################################################################
    
    import prospect.io.read_results as reader
    results_type = "emcee" # | "dynesty"                                                        
    # grab results (dictionary), the obs dictionary, and our corresponding models               
    # When using parameter files set `dangerous=True`                                           
    #tem_name = hfile[:-8]+'{}.h5'
    #result, obs, _ = reader.results_from(tem_name.format(results_type), dangerous=False)
    result, obs, _ = reader.results_from(hfile, dangerous=False)
    #The following commented lines reconstruct the model and sps object,                            # if a parameter file continaing the `build_*` methods was saved along with the results     
    #model = reader.get_model(result)                                                           
    #sps = reader.get_sps(result)                                                               
    # let's look at what's stored in the `result` dictionary                                    
    #print(result.keys())

   # Make plot of data and model
   
    if results_type == "emcee":
        chosen = np.random.choice(result["run_params"]["nwalkers"], size=10, replace=False)
        tracefig = reader.traceplot(result, figsize=(20,10), chains=chosen)
    else:
        tracefig = reader.traceplot(result, figsize=(20,10))
    #plt.savefig(path+'/trace.png')
    # maximum a posteriori (of the locations visited by the MCMC sampler)                           
    imax = np.argmax(result['lnprobability'])
    if results_type == "emcee":
        i, j = np.unravel_index(imax, result['lnprobability'].shape)
        theta_max = result['chain'][i, j, :].copy()
        thin = 5
    else:
        theta_max = result["chain"][imax, :]
        thin = 1

    #f.Write('Optimization value: {}'.format(theta_best))
    #fe.write(str('MAP value: {}'.format(theta_max)))
    #fe.close()
    
    cornerfig = reader.subcorner(result, start=0, thin=thin, truths=theta_best,
                             fig=subplots(15,15,figsize=(27,27))[0])
    #plt.savefig(path+'/corner.png')
    # --- Draw final fitting curve ---

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
    chi = 0
    for i in range(len(wphot)):
	var = ob_unc[i]
        chi += (mphot_map[i] - ob[i])**2/var**2

    # Make plot of best fit model                                                               
    loglog(wspec, mspec_map, label='Model spectrum (MAP)',lw=0.7, color='green', alpha=0.7)
    errorbar(wphot, mphot_map, label='Model photometry (MAP)',marker='s', markersize=10,
             alpha=0.8, ls='', lw=3, markerfacecolor='none', markeredgecolor='green',
             markeredgewidth=3)
    errorbar(wphot, ob, yerr=ob_unc, label='Observed photometry',
             ecolor='red', marker='o', markersize=10, ls='', lw=3, alpha=0.8,
             markerfacecolor='none', markeredgecolor='red', markeredgewidth=3)
    text(0, 1,'Chi-squared/Nphot ='+ str(chi/len(wphot)),
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
    plt.savefig(path+'_fitting.png')
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
    
    #os.remove(hfile)
