#Example script on how to create compound correctionlib objects

import numpy as np
import correctionlib as cl
import uproot
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import pandas as pd
import correctionlib.schemav2 as cs
import functools
import os

def name_parser(name: str) -> dict:
    #Helper function to retrieve correction properties from the name of the histogram stored at the root file 
    low_bin_end = name.find('to')
    high_bin_end = name.find('_', low_bin_end)
    out_dict = {
        "comp": name[3],
        "type": ''.join([the_char for the_char in name[high_bin_end:] if the_char.isalpha()]).lower(),
        "low_eta_bin": float(name[5:low_bin_end].replace('p', '.')),
        "high_eta_bin": float(name[low_bin_end+2:high_bin_end].replace('p', '.')),
        'full_reg_str': name[:-2]
    }
    return out_dict

def make_quantile_mapping(raw_ip_val,
                              ip_bins,
                              mc_cdf,
                              data_cdf,
                              only_formula=True):
        # Find the bin that contains ip_val
        ip_val = np.clip(raw_ip_val,ip_bins[0],ip_bins[-1])
        
        idx = np.searchsorted(ip_bins, ip_val, side="left")
        idx = np.clip(idx, 0, len(mc_cdf) - 1)
        # Get correspondent mc cdf value
        k_cdf2ip = (mc_cdf[idx] - mc_cdf[idx-1]) / (ip_bins[idx]-ip_bins[idx-1])
        # Calculate mc cdf value corresponding to the ip_mc_val using linear approximation
        mc_cdf_val = mc_cdf[idx-1] + k_cdf2ip*(ip_val-ip_bins[idx-1])
        mc_formula_str = f'{mc_cdf[idx-1]} + {k_cdf2ip}*(x-({ip_bins[idx-1]}))'
        # Now find bin of data_cdf that has the same value as mc_cdf_val
        idy = np.searchsorted(data_cdf, mc_cdf_val, side="left")
        idy = np.clip(idy, 0, len(data_cdf) - 2)
        num = (ip_bins[idy] - ip_bins[idy-1])
        den = (data_cdf[idy] - data_cdf[idy-1])
        if (den!=0):
            k_ip2cdf = (ip_bins[idy] - ip_bins[idy-1])/(data_cdf[idy] - data_cdf[idy-1])
            ip_val_corr = ip_bins[idy-1] + k_ip2cdf*(mc_cdf_val - data_cdf[idy-1])
            full_formula = f'{ip_bins[idy-1]} + {k_ip2cdf}*({mc_formula_str} - {data_cdf[idy-1]})'
        if only_formula:
            if (idx==0) or (idx == len(mc_cdf) - 1) or (den==0):
                return 'x'
            else:
                return full_formula
        else: 
            return full_formula, ip_val_corr, {'idx': idx, 'idy':idy, 'mc_cdf_val': mc_cdf_val}

def get_ip_correction_func(h_data, h_mc):
    bins = h_mc.axis().edges()
    mc_vals = h_mc.values()
    mc_vals[0] = 0
    mc_vals[-1] = 0 
    data_vals = h_data.values()
    data_vals[0] = 0 
    data_vals[-1] = 0
    mc_cdf = gaussian_filter1d(np.cumsum(mc_vals)/np.sum(mc_vals), sigma=1)
    data_cdf = gaussian_filter1d(
        np.cumsum(data_vals)/np.sum(data_vals), sigma=1)
    import functools
    formula_list = list(map(functools.partial(make_quantile_mapping,
                                              ip_bins=bins,
                                              mc_cdf=mc_cdf,
                                              data_cdf=data_cdf), bins[:-1]))
    return formula_list, bins

def main():
    #Update here the eras and path
    era_list =['Run3_2022', 'Run3_2022EE', 'Run3_2023','Run3_2023BPix']
    dataset_list = ['mm','ee']
    path =os.getcwd()+'/data/{era}/ip_corrections_{dataset}.root'
    
    for era in era_list:
        corr_list = []
        for dataset in dataset_list:
            the_path = path.format(era=era, dataset=dataset)
            the_input = uproot.open(the_path)
            print(f'Calculating correction for:{the_path}')
            h_list =[]
            for the_name, the_hist in the_input.items():
                h_props = name_parser(the_name) 
                h_props['hist'] = the_hist
                h_list.append(h_props)
            search_hist = lambda h_list, field, val: [h for h in h_list if h[field]==val]

            eta_bin_edges = np.unique([h[bin]
                                    for h in h_list 
                                    for bin in['low_eta_bin','high_eta_bin']])
            low_eta_edges =  np.unique([h['low_eta_bin'] for h in h_list]) 
            components = ['x','y','z']

            corr_binned_by_eta = []
            for the_eta_bin in low_eta_edges:
                #Init empty list to store corrections into categories formed by eta 
                single_eta_bin = []
                for the_comp in components:
                    hists = search_hist(
                        search_hist(
                            h_list,'comp',the_comp),
                        'low_eta_bin',the_eta_bin)
                    h_mc    = search_hist(hists, 'type', 'mc')[0]
                    h_data  = search_hist(hists, 'type', 'data')[0]
                    low_eta = h_mc['low_eta_bin']
                    high_eta = h_mc['high_eta_bin']
                    print(f'Performin quantile mapping for eta in [{low_eta} {high_eta}], component: {the_comp}')
                    #Prepare formulas and bins of ip
                    formula_strings, ip_bins = get_ip_correction_func(h_data['hist'], h_mc['hist'])
                    #Init empty dictionary to store correction formulas
                    cs_formula_list = []
                    #Iterate over the list of formulas
                    for the_formula in formula_strings: 
                        #Fill the list with cf.Formula objects based on the formula strings
                        cs_formula_list.append(
                            cs.Formula(
                            nodetype="formula",
                            variables=["ip"],
                            parser="TFormula",
                            expression=the_formula,
                                ))
                    #When all ip conponents are done fill cs.CategoryItem with list of list if formulas created above
                    #For each eta we have x, y, and z component
                    single_eta_bin.append(
                        cs.CategoryItem(
                            key=the_comp,
                            value=cs.Binning(
                                nodetype="binning",
                                input="ip",
                                edges=ip_bins,
                                content=cs_formula_list,
                                flow="error"
                            )))
                #When full set of categories for a single eta bin us ready, fill the final list with cs.Category containing 
                #3 ip components and for each ip component n_bins of ip value 
                corr_binned_by_eta.append(
                    cs.Category(
                        nodetype="category",
                        input="ip_component",
                        content=single_eta_bin,
                    )
                )
            #fill the correction list (you can store more than one correction i.e. ee and mumu like here
            corr_list.append(
                cs.Correction(
                    name=f"ip_correction_{dataset}",
                    description=f"Impact parameter correction for {era} era, using {dataset} dataset",
                    version=2,
                    inputs=[
                        cs.Variable(name="ip", type="real",description="Impact parameter component"),
                        cs.Variable(name="ip_component", type="string", description="x,y, or z"),
                        cs.Variable(name="eta", type="real", description="Absolute value of peudorapidity of the object for which IP was calculated"),
                    ],
                    output=cs.Variable(name="corrected_ip", type="real", description="Corrected Impact parameter component"),
                    data=cs.Binning(
                        nodetype="binning",
                        input="eta",
                        edges=eta_bin_edges.tolist(),
                        content=corr_binned_by_eta,
                        flow='error',
                        ),
                    )
            )
        #Create correctionSet object containing all your corrections
        cset = cs.CorrectionSet(
            schema_version=2,
            description="Impact paramter correction calculated via quantile mapping",
            corrections=corr_list
        )
        #Dump it into json
        with open(os.getcwd() + f"/output/ip_correction_{era}.json", "w+") as fout:
            fout.write(cset.model_dump_json(exclude_unset=True))

if __name__ == '__main__':
    main()