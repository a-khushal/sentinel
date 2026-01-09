const hre = require("hardhat");

async function main() {
  console.log("Deploying SENTINEL contracts...");

  const ThreatLedger = await hre.ethers.getContractFactory("ThreatLedger");
  const threatLedger = await ThreatLedger.deploy();
  await threatLedger.waitForDeployment();
  
  const threatLedgerAddress = await threatLedger.getAddress();
  console.log("ThreatLedger deployed to:", threatLedgerAddress);

  const FederatedGovernance = await hre.ethers.getContractFactory("FederatedGovernance");
  const federatedGovernance = await FederatedGovernance.deploy();
  await federatedGovernance.waitForDeployment();
  
  const governanceAddress = await federatedGovernance.getAddress();
  console.log("FederatedGovernance deployed to:", governanceAddress);

  console.log("\nDeployment complete!");
  console.log("-------------------");
  console.log("ThreatLedger:", threatLedgerAddress);
  console.log("FederatedGovernance:", governanceAddress);
  
  return {
    threatLedger: threatLedgerAddress,
    federatedGovernance: governanceAddress
  };
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });

