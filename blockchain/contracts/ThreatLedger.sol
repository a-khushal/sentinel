// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract ThreatLedger {
    enum ThreatType { DGA, C2, TUNNEL, UNKNOWN }
    
    struct ThreatReport {
        bytes32 domainHash;
        ThreatType threatType;
        uint256 confidence;
        uint256 timestamp;
        address reporter;
        bytes32 evidenceHash;
    }
    
    struct DomainReputation {
        uint256 totalReports;
        uint256 maliciousScore;
        uint256 firstSeen;
        uint256 lastSeen;
        address[] reporters;
    }
    
    mapping(bytes32 => ThreatReport[]) public threatReports;
    mapping(bytes32 => DomainReputation) public domainReputations;
    mapping(address => bool) public registeredNodes;
    mapping(address => uint256) public nodeStakes;
    mapping(address => uint256) public nodeReportCount;
    
    bytes32[] public reportedDomains;
    
    address public owner;
    uint256 public minStake = 0.01 ether;
    uint256 public totalReports;
    
    event NodeRegistered(address indexed node, uint256 stake);
    event NodeUnregistered(address indexed node);
    event ThreatReported(
        bytes32 indexed domainHash,
        ThreatType threatType,
        uint256 confidence,
        address indexed reporter
    );
    event ReputationUpdated(
        bytes32 indexed domainHash,
        uint256 totalReports,
        uint256 maliciousScore
    );
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    modifier onlyRegisteredNode() {
        require(registeredNodes[msg.sender], "Node not registered");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    function registerNode() external payable {
        require(!registeredNodes[msg.sender], "Already registered");
        require(msg.value >= minStake, "Insufficient stake");
        
        registeredNodes[msg.sender] = true;
        nodeStakes[msg.sender] = msg.value;
        
        emit NodeRegistered(msg.sender, msg.value);
    }
    
    function unregisterNode() external onlyRegisteredNode {
        uint256 stake = nodeStakes[msg.sender];
        
        registeredNodes[msg.sender] = false;
        nodeStakes[msg.sender] = 0;
        
        payable(msg.sender).transfer(stake);
        
        emit NodeUnregistered(msg.sender);
    }
    
    function reportThreat(
        bytes32 _domainHash,
        ThreatType _threatType,
        uint256 _confidence,
        bytes32 _evidenceHash
    ) external onlyRegisteredNode {
        require(_confidence <= 100, "Confidence must be 0-100");
        
        ThreatReport memory report = ThreatReport({
            domainHash: _domainHash,
            threatType: _threatType,
            confidence: _confidence,
            timestamp: block.timestamp,
            reporter: msg.sender,
            evidenceHash: _evidenceHash
        });
        
        threatReports[_domainHash].push(report);
        
        DomainReputation storage rep = domainReputations[_domainHash];
        
        if (rep.firstSeen == 0) {
            rep.firstSeen = block.timestamp;
            reportedDomains.push(_domainHash);
        }
        
        rep.totalReports++;
        rep.maliciousScore += _confidence * nodeStakes[msg.sender] / 1 ether;
        rep.lastSeen = block.timestamp;
        rep.reporters.push(msg.sender);
        
        nodeReportCount[msg.sender]++;
        totalReports++;
        
        emit ThreatReported(_domainHash, _threatType, _confidence, msg.sender);
        emit ReputationUpdated(_domainHash, rep.totalReports, rep.maliciousScore);
    }
    
    function queryReputation(bytes32 _domainHash) external view returns (
        uint256 _totalReports,
        uint256 _maliciousScore,
        uint256 _firstSeen,
        uint256 _lastSeen,
        uint256 _reporterCount
    ) {
        DomainReputation storage rep = domainReputations[_domainHash];
        return (
            rep.totalReports,
            rep.maliciousScore,
            rep.firstSeen,
            rep.lastSeen,
            rep.reporters.length
        );
    }
    
    function getThreatReports(bytes32 _domainHash) external view returns (ThreatReport[] memory) {
        return threatReports[_domainHash];
    }
    
    function getReportedDomainsCount() external view returns (uint256) {
        return reportedDomains.length;
    }
    
    function getReportedDomains(uint256 _start, uint256 _count) external view returns (bytes32[] memory) {
        uint256 end = _start + _count;
        if (end > reportedDomains.length) {
            end = reportedDomains.length;
        }
        
        bytes32[] memory result = new bytes32[](end - _start);
        for (uint256 i = _start; i < end; i++) {
            result[i - _start] = reportedDomains[i];
        }
        
        return result;
    }
    
    function isNodeRegistered(address _node) external view returns (bool) {
        return registeredNodes[_node];
    }
    
    function getNodeStats(address _node) external view returns (
        bool registered,
        uint256 stake,
        uint256 reports
    ) {
        return (
            registeredNodes[_node],
            nodeStakes[_node],
            nodeReportCount[_node]
        );
    }
    
    function setMinStake(uint256 _minStake) external onlyOwner {
        minStake = _minStake;
    }
    
    function hashDomain(string memory _domain) external pure returns (bytes32) {
        return keccak256(abi.encodePacked(_domain));
    }
}

