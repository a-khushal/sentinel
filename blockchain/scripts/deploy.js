const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const network = hre.network.name;
  console.log(`Deploying SENTINEL contracts to ${network}...`);
  console.log("");

  const [deployer] = await hre.ethers.getSigners();
  console.log("Deployer:", deployer.address);
  
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Balance:", hre.ethers.formatEther(balance), "ETH");
  console.log("");

  const ThreatLedger = await hre.ethers.getContractFactory("ThreatLedger");
  console.log("Deploying ThreatLedger...");
  const threatLedger = await ThreatLedger.deploy();
  await threatLedger.waitForDeployment();
  const threatLedgerAddress = await threatLedger.getAddress();
  console.log("ThreatLedger deployed to:", threatLedgerAddress);

  const FederatedGovernance = await hre.ethers.getContractFactory("FederatedGovernance");
  console.log("Deploying FederatedGovernance...");
  const federatedGovernance = await FederatedGovernance.deploy();
  await federatedGovernance.waitForDeployment();
  const governanceAddress = await federatedGovernance.getAddress();
  console.log("FederatedGovernance deployed to:", governanceAddress);

  const deployment = {
    network: network,
    chainId: hre.network.config.chainId,
    deployer: deployer.address,
    timestamp: new Date().toISOString(),
    contracts: {
      ThreatLedger: threatLedgerAddress,
      FederatedGovernance: governanceAddress
    }
  };

  const deploymentPath = path.join(__dirname, "..", "deployment.json");
  fs.writeFileSync(deploymentPath, JSON.stringify(deployment, null, 2));
  console.log("");
  console.log("Deployment saved to:", deploymentPath);

  console.log("");
  console.log("=".repeat(50));
  console.log("DEPLOYMENT COMPLETE");
  console.log("=".repeat(50));
  console.log("Network:", network);
  console.log("ThreatLedger:", threatLedgerAddress);
  console.log("FederatedGovernance:", governanceAddress);
  console.log("=".repeat(50));

  if (network === "sepolia") {
    console.log("");
    console.log("View on Etherscan:");
    console.log(`https://sepolia.etherscan.io/address/${threatLedgerAddress}`);
    console.log(`https://sepolia.etherscan.io/address/${governanceAddress}`);
  }

  return deployment;
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
