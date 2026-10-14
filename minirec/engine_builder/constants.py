DATASET_PATH = "s3://rec-engine/data/lavka"
ENGINE_ARCHIVE_PATH = "s3://rec-engine/service/v1/archive.tar.gz"
# point-in-time snapshots built during the ranker/Mixigen replay; never read by engine-downloader
REPLAY_ARCHIVE_PATH = "s3://rec-engine/service/replay/archive.tar.gz"
ENGINE_CONFIG_PATH = "../engine_server/config.yaml"
RANKER_CONFIG_PATH = "ranker_config.yaml"
MIXIGEN_CONFIG_PATH = "mixigen_config.yaml"
