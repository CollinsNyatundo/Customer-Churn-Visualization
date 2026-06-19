# Kubernetes Deployment

## Prerequisites
- `kubectl` configured for your cluster
- `cert-manager` installed (for TLS)
- `nginx-ingress-controller` installed
- Container image pushed to GHCR (done automatically by CD workflow)

## Deploy

```bash
# 1. Create namespace and config
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# 2. Create secrets (fill in real values first)
# Edit k8s/secrets.yaml, base64-encode your values:
#   echo -n "your-api-key" | base64
kubectl apply -f k8s/secrets.yaml

# 3. Deploy services
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/ingress.yaml

# 4. Verify
kubectl get pods -n churn-ml
kubectl get svc -n churn-ml
```

## Scaling
The API deployment includes an HPA (HorizontalPodAutoscaler) that scales
from 2 → 8 replicas when CPU exceeds 70%.

## Updating
The CD workflow auto-pushes a new image on every merge to `main`.
To roll out the new image:
```bash
kubectl rollout restart deployment/churn-api -n churn-ml
kubectl rollout status deployment/churn-api -n churn-ml
```
