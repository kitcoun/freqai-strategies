# FreqAI strategies

## QuickAdapter

### Quick start

Change the timezone according to your location in [`docker-compose.yml`](./quickadapter/docker-compose.yml).

From the repository root, configure, build and start the QuickAdapter container:

```shell
cd quickadapter
cp user_data/config-template.json user_data/config.json
```

Adapt the configuration to your needs: edit `user_data/config.json` to set your exchange API keys and tune the `freqai` section.

Then build and start the container:

```shell
docker compose up -d --build
```

## ReforceXY

### Quick start

Change the timezone according to your location in [`docker-compose.yml`](./ReforceXY/docker-compose.yml).

From the repository root, configure, build and start the ReforceXY container:

```shell
cd ReforceXY
cp user_data/config-template.json user_data/config.json
```

Adapt the configuration to your needs: edit `user_data/config.json` to set your exchange API keys and tune the `freqai` section.

Then build and start the container:

```shell
docker compose up -d --build
```

[Reward Space Analysis](./ReforceXY/reward_space_analysis/README.md)

## Common workflows

List running compose services and the containers they created:

```shell
docker compose ps
```

Enter a running service:

```shell
# use the compose service name (e.g. "freqtrade")
docker compose exec freqtrade /bin/sh
```

View logs:

```shell
# service logs (compose maps service -> container(s))
docker compose logs -f freqtrade

# or follow a specific container's logs
docker logs -f freqtrade-quickadapter
```

Stop and remove the compose stack:

```shell
docker compose down
```

---

## Note

> Do not expect any support of any kind on the Internet. Nevertheless, PRs implementing documentation, bug fixes, cleanups or sensible features will be discussed and might get merged.

