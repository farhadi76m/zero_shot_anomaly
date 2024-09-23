# Function to set up the Mask2Former model configuration and load the weights
def setup_mask2former(cfg_path, model_path):
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    """
    Sets up Mask2Former model configuration.
    """
    args = edict({'config_file': cfg_path, 'eval-only': True, 'opts': ["MODEL.WEIGHTS", model_path]})

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg, DefaultPredictor(cfg)

# Function to load SAM model (once)
def setup_sam(sam_checkpoint):
    from segment_anything import SamPredictor, sam_model_registry

    """
    Sets up the SAM model.
    """
    sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint)
    sam.to("cuda")
    return SamPredictor(sam)

def setup_oneformer(dataset, model_path, use_swin):
    
    from demo.defaults import DefaultPredictor
    # import OneFormer Project
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from demo.defaults import DefaultPredictor

    from oneformer import (
        add_oneformer_config,
        add_common_config,
        add_swin_config,
        add_dinat_config,
        add_convnext_config,
    )
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file("/content/OneFormer/configs/cityscapes/convnext/oneformer_convnext_xlarge_bs16_90k.yaml")
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = "/content/mapillary_pretrain_250_16_convnext_xl_oneformer_cityscapes_90k.pth"
    cfg.freeze()
    
   
    predictor = DefaultPredictor(cfg)
    
    return cfg, predictor