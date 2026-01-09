// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract FederatedGovernance {
    struct ModelProposal {
        bytes32 modelHash;
        string ipfsUri;
        uint256 epoch;
        uint256 votesFor;
        uint256 votesAgainst;
        uint256 proposedAt;
        address proposer;
        bool executed;
        mapping(address => bool) hasVoted;
    }
    
    struct ModelVersion {
        bytes32 modelHash;
        string ipfsUri;
        uint256 epoch;
        uint256 timestamp;
        address proposer;
    }
    
    mapping(uint256 => ModelProposal) public proposals;
    mapping(address => bool) public registeredNodes;
    mapping(address => uint256) public nodeStakes;
    
    ModelVersion[] public modelHistory;
    
    uint256 public proposalCount;
    uint256 public currentEpoch;
    uint256 public votingPeriod = 1 hours;
    uint256 public quorumPercentage = 51;
    uint256 public minStake = 0.01 ether;
    
    address public owner;
    
    event ProposalCreated(
        uint256 indexed proposalId,
        bytes32 modelHash,
        string ipfsUri,
        uint256 epoch,
        address indexed proposer
    );
    event Voted(
        uint256 indexed proposalId,
        address indexed voter,
        bool support,
        uint256 weight
    );
    event ProposalExecuted(
        uint256 indexed proposalId,
        bytes32 modelHash,
        uint256 epoch
    );
    event NodeRegistered(address indexed node, uint256 stake);
    
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
    
    function proposeModelUpdate(
        bytes32 _modelHash,
        string memory _ipfsUri
    ) external onlyRegisteredNode returns (uint256) {
        uint256 proposalId = proposalCount++;
        
        ModelProposal storage proposal = proposals[proposalId];
        proposal.modelHash = _modelHash;
        proposal.ipfsUri = _ipfsUri;
        proposal.epoch = currentEpoch + 1;
        proposal.proposedAt = block.timestamp;
        proposal.proposer = msg.sender;
        proposal.executed = false;
        
        emit ProposalCreated(proposalId, _modelHash, _ipfsUri, proposal.epoch, msg.sender);
        
        return proposalId;
    }
    
    function vote(uint256 _proposalId, bool _support) external onlyRegisteredNode {
        ModelProposal storage proposal = proposals[_proposalId];
        
        require(!proposal.executed, "Already executed");
        require(!proposal.hasVoted[msg.sender], "Already voted");
        require(block.timestamp <= proposal.proposedAt + votingPeriod, "Voting ended");
        
        proposal.hasVoted[msg.sender] = true;
        
        uint256 weight = nodeStakes[msg.sender];
        
        if (_support) {
            proposal.votesFor += weight;
        } else {
            proposal.votesAgainst += weight;
        }
        
        emit Voted(_proposalId, msg.sender, _support, weight);
    }
    
    function executeProposal(uint256 _proposalId) external {
        ModelProposal storage proposal = proposals[_proposalId];
        
        require(!proposal.executed, "Already executed");
        require(block.timestamp > proposal.proposedAt + votingPeriod, "Voting not ended");
        
        uint256 totalVotes = proposal.votesFor + proposal.votesAgainst;
        require(totalVotes > 0, "No votes");
        
        uint256 forPercentage = (proposal.votesFor * 100) / totalVotes;
        require(forPercentage >= quorumPercentage, "Quorum not reached");
        
        proposal.executed = true;
        currentEpoch = proposal.epoch;
        
        modelHistory.push(ModelVersion({
            modelHash: proposal.modelHash,
            ipfsUri: proposal.ipfsUri,
            epoch: proposal.epoch,
            timestamp: block.timestamp,
            proposer: proposal.proposer
        }));
        
        emit ProposalExecuted(_proposalId, proposal.modelHash, proposal.epoch);
    }
    
    function getCurrentModel() external view returns (
        bytes32 modelHash,
        string memory ipfsUri,
        uint256 epoch,
        uint256 timestamp
    ) {
        if (modelHistory.length == 0) {
            return (bytes32(0), "", 0, 0);
        }
        
        ModelVersion storage current = modelHistory[modelHistory.length - 1];
        return (current.modelHash, current.ipfsUri, current.epoch, current.timestamp);
    }
    
    function getModelHistory(uint256 _start, uint256 _count) external view returns (ModelVersion[] memory) {
        uint256 end = _start + _count;
        if (end > modelHistory.length) {
            end = modelHistory.length;
        }
        
        ModelVersion[] memory result = new ModelVersion[](end - _start);
        for (uint256 i = _start; i < end; i++) {
            result[i - _start] = modelHistory[i];
        }
        
        return result;
    }
    
    function getProposalInfo(uint256 _proposalId) external view returns (
        bytes32 modelHash,
        string memory ipfsUri,
        uint256 epoch,
        uint256 votesFor,
        uint256 votesAgainst,
        uint256 proposedAt,
        address proposer,
        bool executed
    ) {
        ModelProposal storage p = proposals[_proposalId];
        return (
            p.modelHash,
            p.ipfsUri,
            p.epoch,
            p.votesFor,
            p.votesAgainst,
            p.proposedAt,
            p.proposer,
            p.executed
        );
    }
    
    function setVotingPeriod(uint256 _period) external onlyOwner {
        votingPeriod = _period;
    }
    
    function setQuorum(uint256 _percentage) external onlyOwner {
        require(_percentage <= 100, "Invalid percentage");
        quorumPercentage = _percentage;
    }
}

