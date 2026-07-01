def tensorboard_log_image(log_writer, tag: str, image_tensor, step):
    log_writer.experiment.add_image(
        tag,
        image_tensor,
        step,
    )

def wandb_log_image(log_writer, tag: str, image_tensor, step):
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging was requested, but wandb is not installed. "
            "Install wandb or set logger_config.logger to tensorboard."
        ) from exc

    image_dict = {
        tag: wandb.Image(image_tensor),
    }
    log_writer.experiment.log(
        image_dict,
        step=step,
    )
