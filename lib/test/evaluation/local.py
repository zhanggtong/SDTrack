from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/muyh/tracking_datasets/got10k_lmdb'
    settings.got10k_path = '/home/muyh/tracking_datasets/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/muyh/tracking_datasets/itb'
    settings.lasot_extension_subset_path = '/home/muyh/tracking_datasets/LaSOT_extension_subset'
    settings.lasot_lmdb_path = '/home/muyh/tracking_datasets/lasot_lmdb'
    settings.lasot_path = '/home/muyh/tracking_datasets/lasot'
    settings.mgit_path = '/home/muyh/tracking_datasets/mgit'
    settings.network_path = '/home/muyh/ysr/tlot_01/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/muyh/tracking_datasets/nfs'
    settings.otb_path = '/home/muyh/tracking_datasets/OTB_sentences'
    settings.prj_dir = '/home/zhanggt/SDTrack'
    settings.result_plot_path = '/home/zhanggt/SDTrack/output/test/result_plots'
    settings.results_path = '/home/zhanggt/SDTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/zhanggt/SDTrack/output'
    settings.segmentation_path = '/home/zhanggt/SDTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/muyh/tracking_datasets/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/muyh/tracking_datasets/t2'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/muyh/tracking_datasets/trackingnet'
    settings.uav_path = '/home/muyh/tracking_datasets/uav'
    settings.vot18_path = '/home/muyh/tracking_datasets/vot2018'
    settings.vot22_path = '/home/muyh/tracking_datasets/vot2022'
    settings.vot_path = '/home/muyh/tracking_datasets/VOT2019'
    settings.youtubevos_dir = ''

    return settings

