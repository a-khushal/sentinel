from scapy.all import sniff, DNS, DNSQR, DNSRR, IP, UDP
from dataclasses import dataclass
from typing import Callable, Optional
from datetime import datetime
import threading
import queue

@dataclass
class DNSQuery:
    timestamp: datetime
    src_ip: str
    dst_ip: str
    query_name: str
    query_type: int
    response_code: Optional[int] = None
    answers: Optional[list] = None
    ttl: Optional[int] = None

class DNSSniffer:
    def __init__(self, interface: str = "eth0", callback: Optional[Callable] = None):
        self.interface = interface
        self.callback = callback
        self.packet_queue = queue.Queue()
        self.running = False
        self._thread = None
    
    def _process_packet(self, packet):
        if not packet.haslayer(DNS):
            return
        
        dns = packet[DNS]
        ip = packet[IP] if packet.haslayer(IP) else None
        
        if not ip:
            return
        
        query = DNSQuery(
            timestamp=datetime.now(),
            src_ip=ip.src,
            dst_ip=ip.dst,
            query_name=dns[DNSQR].qname.decode() if dns.haslayer(DNSQR) else "",
            query_type=dns[DNSQR].qtype if dns.haslayer(DNSQR) else 0,
            response_code=dns.rcode if dns.qr == 1 else None,
            answers=self._extract_answers(dns) if dns.qr == 1 else None,
            ttl=self._extract_ttl(dns) if dns.qr == 1 else None
        )
        
        if query.query_name:
            query.query_name = query.query_name.rstrip('.')
        
        self.packet_queue.put(query)
        
        if self.callback:
            self.callback(query)
    
    def _extract_answers(self, dns) -> list:
        answers = []
        for i in range(dns.ancount):
            if dns.haslayer(DNSRR):
                rr = dns[DNSRR][i] if i < dns.ancount else None
                if rr:
                    answers.append(str(rr.rdata))
        return answers
    
    def _extract_ttl(self, dns) -> Optional[int]:
        if dns.haslayer(DNSRR):
            return dns[DNSRR].ttl
        return None
    
    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self._thread.start()
    
    def _sniff_loop(self):
        sniff(
            iface=self.interface,
            filter="udp port 53",
            prn=self._process_packet,
            store=False,
            stop_filter=lambda _: not self.running
        )
    
    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
    
    def get_queries(self, max_items: int = 100) -> list:
        queries = []
        while not self.packet_queue.empty() and len(queries) < max_items:
            try:
                queries.append(self.packet_queue.get_nowait())
            except queue.Empty:
                break
        return queries

