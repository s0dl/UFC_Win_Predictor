# Cloud Run Terraform

This Terraform deploys one Cloud Run service:

- `ufc-app` - FastAPI backend serving the Vite frontend and API from one container

The app exposes:

- `/` - frontend
- `/api/health` and `/health`
- `/api/fighters` and `/fighters`
- `/api/predict` and `/predict`

## Build And Push Image

Create an Artifact Registry Docker repository once:

```bash
gcloud artifacts repositories create ufc \
  --repository-format=docker \
  --location=us-central1
```

Build and push:

```bash
PROJECT_ID=your-gcp-project-id
REGION=us-central1
REPO=$REGION-docker.pkg.dev/$PROJECT_ID/ufc
TAG=$(date +%Y%m%d%H%M%S)

gcloud auth configure-docker $REGION-docker.pkg.dev

docker build -f ../../Dockerfile -t $REPO/ufc-app:$TAG ../..
docker push $REPO/ufc-app:$TAG
```

## Deploy

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
project_id = "your-gcp-project-id"
region     = "us-central1"
image      = "us-central1-docker.pkg.dev/your-gcp-project-id/ufc/ufc-app:YOUR_TAG"
```

Then deploy:

```bash
terraform init
terraform plan
terraform apply
```

Terraform outputs the app URL.

## Migrating From The Old Two-Service Terraform

If your local Terraform state still contains `google_cloud_run_v2_service.api`
and `google_cloud_run_v2_service.frontend`, destroy the old stack before
applying this one:

```bash
terraform destroy
```

Then apply the one-service config. If you already deleted the old services
manually, remove them from local state:

```bash
terraform state rm google_cloud_run_v2_service.api
terraform state rm google_cloud_run_v2_service.frontend
terraform state rm google_cloud_run_v2_service_iam_member.api_public
terraform state rm google_cloud_run_v2_service_iam_member.frontend_public
```
