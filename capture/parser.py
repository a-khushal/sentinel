from scapy.all import rdpcap, DNS, DNSQR, DNSRR, IP
from dataclasses import dataclass
from typing import List, Generator
from datetime import datetime
from .sniffer import DNSQuery
import os

class DNSParser:
    def __init__(self):
        self.queries: List[DNSQuery] = []
    
    def parse_pcap(self, pcap_path: str) -> List[DNSQuery]:
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
        
        packets = rdpcap(pcap_path)
        queries = []
        
        for packet in packets:
            if not packet.haslayer(DNS):
                continue
            
            dns = packet[DNS]
            ip = packet[IP] if packet.haslayer(IP) else None
            
            if not ip or not dns.haslayer(DNSQR):
                continue
            
            query_name = dns[DNSQR].qname.decode() if dns[DNSQR].qname else ""
            query_name = query_name.rstrip('.')
            
            answers = []
            ttl = None
            if dns.qr == 1 and dns.haslayer(DNSRR):
                for i in range(min(dns.ancount, 10)):
                    try:
                        rr = dns[DNSRR][i]
                        answers.append(str(rr.rdata))
                        if ttl is None:
                            ttl = rr.ttl
                    except:
                        break
            
            query = DNSQuery(
                timestamp=datetime.fromtimestamp(float(packet.time)),
                src_ip=ip.src,
                dst_ip=ip.dst,
                query_name=query_name,
                query_type=dns[DNSQR].qtype,
                response_code=dns.rcode if dns.qr == 1 else None,
                answers=answers if answers else None,
                ttl=ttl
            )
            
            queries.append(query)
        
        self.queries = queries
        return queries
    
    def parse_pcap_stream(self, pcap_path: str, batch_size: int = 1000) -> Generator[List[DNSQuery], None, None]:
        packets = rdpcap(pcap_path)
        batch = []
        
        for packet in packets:
            if not packet.haslayer(DNS):
                continue
            
            dns = packet[DNS]
            ip = packet[IP] if packet.haslayer(IP) else None
            
            if not ip or not dns.haslayer(DNSQR):
                continue
            
            query_name = dns[DNSQR].qname.decode() if dns[DNSQR].qname else ""
            query_name = query_name.rstrip('.')
            
            query = DNSQuery(
                timestamp=datetime.fromtimestamp(float(packet.time)),
                src_ip=ip.src,
                dst_ip=ip.dst,
                query_name=query_name,
                query_type=dns[DNSQR].qtype,
                response_code=dns.rcode if dns.qr == 1 else None,
                answers=None,
                ttl=None
            )
            
            batch.append(query)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
    
    def get_unique_domains(self) -> List[str]:
        return list(set(q.query_name for q in self.queries if q.query_name))
    
    def get_unique_clients(self) -> List[str]:
        return list(set(q.src_ip for q in self.queries))

