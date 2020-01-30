The models in this directory are usable in audio mode with the `similarity_to_quality_model` flag.

These models were trained on the CoreSV15, AACvOpus14, and TCDAudio14 datasets.

libsvm_nu_svr_model.txt:
This is the 'default' model, and is the same as tcdaudio14_aacvopus_coresv_svrnsim_n.68_g.01_c1.model.

tcdaudio14_aacvopus_coresv_svrnsim_n.68_g.01_c1.model:
After discovering that 'tcdaudio_aacvopus_coresv_grid_nu0.3_c126.237786175_g0.204475514639.model'  was too specific and had non-monotonic behavior for out of distribution data, this model was trained with a weaker gamma and cost parameters (c=1, gamma=.01).  This makes the projection from NSIM to a higher dimension more smooth, and less likely to have non-monotonic regions (since the general trend of NSIM vs MOS is positively correlated.  This also means that it will have a higher cross validation error (0.43703 MSE) but it should be a more general model, which is why it is used for the default in v3.00.

tcdaudio_aacvopus_coresv_grid_nu0.3_c126.237786175_g0.204475514639.model:
This model used a fine grid search to minimize the mean-squared-error in the cross validation set (it was .220864 on the cross validation set).  It should work reasonably well for data like the train set (full-band 24kHz bandwidth, 24 kbps or higher).  However, it was found to exhibit non-monotonic behavior for out-of-distribution data, such as narrowband or wideband for very low bitrates (< 24 kbps), meaning that the similarity index would increase, but the MOS would go down.

tcdvoip_nu.568_c5.31474325639_g3.17773760038_model.txt
This is an experimental model for speech trained on the TCDVoip dataset, for use with (wideband, 16kHz) speech mode.  It requires a (relatively simple) code change to use FVNSIM and the SVR mapper instead of the polynomial mapper.
