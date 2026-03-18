"""Simple HTTP server with CORS headers for serving .splat files via Cloudflare tunnel."""
import http.server
import os
import sys

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Range')
        self.send_header('Access-Control-Expose-Headers', 'Content-Length, Content-Range')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    print(f"Serving {directory} on port {port} with CORS enabled")
    server = http.server.HTTPServer(('', port), CORSHandler)
    server.serve_forever()
