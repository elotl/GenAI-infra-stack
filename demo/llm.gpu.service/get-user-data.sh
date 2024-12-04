clustername=$1
region=$2
eksctl get cluster --region $region --name $clustername -o json \
   | jq --raw-output '.[] | "settings.kubernetes.api-server = \"" + .Endpoint + "\"\nsettings.kubernetes.cluster-certificate =\"" + .CertificateAuthority.Data + "\"\n"' > user-data.toml
