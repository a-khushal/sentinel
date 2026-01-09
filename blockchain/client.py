from web3 import Web3
from eth_account import Account
from typing import Dict, List, Optional, Tuple
import json
import hashlib
import os
from dataclasses import dataclass
from enum import IntEnum

BLOCKCHAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOYMENT_FILE = os.path.join(BLOCKCHAIN_DIR, "deployment.json")

class ThreatType(IntEnum):
    DGA = 0
    C2 = 1
    TUNNEL = 2
    UNKNOWN = 3

@dataclass
class ThreatReport:
    domain_hash: str
    threat_type: ThreatType
    confidence: int
    timestamp: int
    reporter: str
    evidence_hash: str

@dataclass
class DomainReputation:
    total_reports: int
    malicious_score: int
    first_seen: int
    last_seen: int
    reporter_count: int

THREAT_LEDGER_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "inputs": [],
        "name": "registerNode",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "unregisterNode",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "_domainHash", "type": "bytes32"},
            {"name": "_threatType", "type": "uint8"},
            {"name": "_confidence", "type": "uint256"},
            {"name": "_evidenceHash", "type": "bytes32"}
        ],
        "name": "reportThreat",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "_domainHash", "type": "bytes32"}],
        "name": "queryReputation",
        "outputs": [
            {"name": "_totalReports", "type": "uint256"},
            {"name": "_maliciousScore", "type": "uint256"},
            {"name": "_firstSeen", "type": "uint256"},
            {"name": "_lastSeen", "type": "uint256"},
            {"name": "_reporterCount", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "_domainHash", "type": "bytes32"}],
        "name": "getThreatReports",
        "outputs": [
            {
                "components": [
                    {"name": "domainHash", "type": "bytes32"},
                    {"name": "threatType", "type": "uint8"},
                    {"name": "confidence", "type": "uint256"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "reporter", "type": "address"},
                    {"name": "evidenceHash", "type": "bytes32"}
                ],
                "name": "",
                "type": "tuple[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "_node", "type": "address"}],
        "name": "isNodeRegistered",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "_node", "type": "address"}],
        "name": "getNodeStats",
        "outputs": [
            {"name": "registered", "type": "bool"},
            {"name": "stake", "type": "uint256"},
            {"name": "reports", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getReportedDomainsCount",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "_start", "type": "uint256"},
            {"name": "_count", "type": "uint256"}
        ],
        "name": "getReportedDomains",
        "outputs": [{"name": "", "type": "bytes32[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "_domain", "type": "string"}],
        "name": "hashDomain",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "pure",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalReports",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "minStake",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]

class BlockchainClient:
    def __init__(self, 
                 rpc_url: str,
                 private_key: str,
                 contract_address: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.account = Account.from_key(private_key)
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=THREAT_LEDGER_ABI
        )
    
    @property
    def address(self) -> str:
        return self.account.address
    
    def is_connected(self) -> bool:
        return self.w3.is_connected()
    
    def get_balance(self) -> float:
        balance_wei = self.w3.eth.get_balance(self.account.address)
        return self.w3.from_wei(balance_wei, 'ether')
    
    def hash_domain(self, domain: str) -> bytes:
        domain = domain.lower().rstrip('.')
        return Web3.keccak(text=domain)
    
    def register_node(self, stake_eth: float = 0.01) -> str:
        stake_wei = self.w3.to_wei(stake_eth, 'ether')
        
        tx = self.contract.functions.registerNode().build_transaction({
            'from': self.account.address,
            'value': stake_wei,
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
        })
        
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt['transactionHash'].hex()
    
    def report_threat(self,
                      domain: str,
                      threat_type: ThreatType,
                      confidence: int,
                      evidence: str = "") -> str:
        domain_hash = self.hash_domain(domain)
        evidence_hash = Web3.keccak(text=evidence) if evidence else bytes(32)
        
        tx = self.contract.functions.reportThreat(
            domain_hash,
            int(threat_type),
            confidence,
            evidence_hash
        ).build_transaction({
            'from': self.account.address,
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
        })
        
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt['transactionHash'].hex()
    
    def query_reputation(self, domain: str) -> DomainReputation:
        domain_hash = self.hash_domain(domain)
        result = self.contract.functions.queryReputation(domain_hash).call()
        
        return DomainReputation(
            total_reports=result[0],
            malicious_score=result[1],
            first_seen=result[2],
            last_seen=result[3],
            reporter_count=result[4]
        )
    
    def get_threat_reports(self, domain: str) -> List[ThreatReport]:
        domain_hash = self.hash_domain(domain)
        reports = self.contract.functions.getThreatReports(domain_hash).call()
        
        return [
            ThreatReport(
                domain_hash=r[0].hex(),
                threat_type=ThreatType(r[1]),
                confidence=r[2],
                timestamp=r[3],
                reporter=r[4],
                evidence_hash=r[5].hex()
            )
            for r in reports
        ]
    
    def is_node_registered(self, address: Optional[str] = None) -> bool:
        addr = address or self.account.address
        return self.contract.functions.isNodeRegistered(addr).call()
    
    def get_node_stats(self, address: Optional[str] = None) -> Dict:
        addr = address or self.account.address
        result = self.contract.functions.getNodeStats(addr).call()
        
        return {
            'registered': result[0],
            'stake': self.w3.from_wei(result[1], 'ether'),
            'reports': result[2]
        }
    
    def get_total_reports(self) -> int:
        return self.contract.functions.totalReports().call()
    
    def get_reported_domains_count(self) -> int:
        return self.contract.functions.getReportedDomainsCount().call()
    
    def get_reported_domains(self, start: int = 0, count: int = 100) -> List[str]:
        domains = self.contract.functions.getReportedDomains(start, count).call()
        return [d.hex() for d in domains]
    
    def get_min_stake(self) -> float:
        stake_wei = self.contract.functions.minStake().call()
        return self.w3.from_wei(stake_wei, 'ether')


def load_deployment() -> Optional[Dict]:
    if os.path.exists(DEPLOYMENT_FILE):
        with open(DEPLOYMENT_FILE, 'r') as f:
            return json.load(f)
    return None

def get_blockchain_client(private_key: Optional[str] = None) -> 'BlockchainClient | MockBlockchainClient':
    deployment = load_deployment()
    
    if deployment is None:
        print("No deployment.json found, using mock client")
        return MockBlockchainClient()
    
    pk = private_key or os.environ.get('PRIVATE_KEY')
    if not pk:
        print("No PRIVATE_KEY found, using mock client")
        return MockBlockchainClient()
    
    network = deployment.get('network', 'sepolia')
    
    rpc_urls = {
        'sepolia': os.environ.get('SEPOLIA_RPC_URL', 'https://rpc.sepolia.org'),
        'amoy': os.environ.get('POLYGON_AMOY_RPC', 'https://rpc-amoy.polygon.technology'),
        'hardhat': 'http://127.0.0.1:8545',
        'localhost': 'http://127.0.0.1:8545',
    }
    
    rpc_url = rpc_urls.get(network, rpc_urls['sepolia'])
    contract_address = deployment['contracts']['ThreatLedger']
    
    try:
        client = BlockchainClient(rpc_url, pk, contract_address)
        if client.is_connected():
            print(f"Connected to {network} blockchain")
            print(f"Contract: {contract_address}")
            return client
        else:
            print(f"Failed to connect to {network}, using mock client")
            return MockBlockchainClient()
    except Exception as e:
        print(f"Blockchain client error: {e}, using mock client")
        return MockBlockchainClient()


class MockBlockchainClient:
    def __init__(self):
        self.threats: Dict[str, List[Dict]] = {}
        self.reputations: Dict[str, DomainReputation] = {}
        self.registered = False
        self.reports_count = 0
        self.mock_mode = True
    
    def is_connected(self) -> bool:
        return True
    
    def hash_domain(self, domain: str) -> str:
        return hashlib.sha256(domain.lower().encode()).hexdigest()
    
    def register_node(self, stake_eth: float = 0.01) -> str:
        self.registered = True
        return "0x" + "0" * 64
    
    def report_threat(self,
                      domain: str,
                      threat_type: ThreatType,
                      confidence: int,
                      evidence: str = "") -> str:
        domain_hash = self.hash_domain(domain)
        
        if domain_hash not in self.threats:
            self.threats[domain_hash] = []
            self.reputations[domain_hash] = DomainReputation(
                total_reports=0,
                malicious_score=0,
                first_seen=0,
                last_seen=0,
                reporter_count=0
            )
        
        self.threats[domain_hash].append({
            'threat_type': threat_type,
            'confidence': confidence,
            'evidence': evidence
        })
        
        rep = self.reputations[domain_hash]
        self.reputations[domain_hash] = DomainReputation(
            total_reports=rep.total_reports + 1,
            malicious_score=rep.malicious_score + confidence,
            first_seen=rep.first_seen or 1,
            last_seen=1,
            reporter_count=rep.reporter_count + 1
        )
        
        self.reports_count += 1
        return "0x" + hashlib.sha256(f"{domain}{confidence}".encode()).hexdigest()
    
    def query_reputation(self, domain: str) -> DomainReputation:
        domain_hash = self.hash_domain(domain)
        return self.reputations.get(domain_hash, DomainReputation(0, 0, 0, 0, 0))
    
    def is_node_registered(self, address: Optional[str] = None) -> bool:
        return self.registered
    
    def get_total_reports(self) -> int:
        return self.reports_count

