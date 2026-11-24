# Chạy server trung tâm bằng Docker

Server của game (`src/server/`) đóng thành một image Docker (chỉ thư viện chuẩn của Python,
~176 MB, chạy bằng user thường uid 10001). `deploy/docker-compose.yml` chạy hai service:
`server` và `backup` (sao lưu SQLite mỗi ngày, giữ 7 bản). Mọi lệnh dưới đây dùng được trong
**fish** (không có heredoc). Chạy từ thư mục gốc của repo trừ khi ghi khác.

Dữ liệu nằm trong 3 volume Docker (không nằm trong image, không nằm trong repo):

| volume | chứa | tên mặc định |
|---|---|---|
| data | `server.db` (tài khoản, ví, điểm) | `hcpd-data` |
| certs | `cert.pem`, `key.pem` | `hcpd-certs` |
| backups | `server-YYYYmmdd-HHMMSS.db` | `hcpd-backups` |

## 1. Lần đầu

```fish
cp deploy/.env.example deploy/.env
bash tools/make_dev_cert.sh ~/hcpd-certs localhost
cd deploy; docker compose build; cd ..
bash deploy/import_certs.sh ~/hcpd-certs
cd deploy; docker compose up -d; cd ..
docker ps --filter name=hcpd
```

`deploy/.env` và khoá `key.pem` không bao giờ vào git. Giữ `~/hcpd-certs/key.pem` bí mật;
chỉ `cert.pem` được gửi cho người chơi. Cert tự ký có hiệu lực 825 ngày; hết hạn thì làm lại
bước cert và gửi `cert.pem` mới cho mọi người chơi (game ghim đúng file này).

Mặc định cổng chỉ mở trên `127.0.0.1` (máy này). Sau ~10 giây `docker ps` phải ghi `(healthy)`.

## 2. Bật / tắt / xem log

```fish
cd deploy
docker compose up -d
docker compose stop
docker compose restart
docker compose logs --tail 50 server
docker compose logs --tail 20 backup
cd ..
```

Server tự chạy lại khi bị kill hoặc khi máy khởi động lại (`restart: unless-stopped`; Docker
phải tự bật cùng máy: Docker Desktop → "Start Docker Desktop when you sign in"; Docker Engine →
`systemctl enable docker`).

Lưu ý: Docker coi `docker kill` và `docker stop` là do bạn tự dừng, nên **không** tự bật lại
(đã đo trên Docker Desktop 29.9.3, cả ba chính sách restart). Dừng bằng tay thì bật lại bằng
`docker compose up -d`. Khi tiến trình server tự chết, hoặc máy khởi động lại, nó tự lên lại.

Quản trị (xem tài khoản, mở khoá, thu hồi phiên) chạy ngay trong container, không qua mạng:

```fish
docker exec hcpd-server python -m src.server.admin --data-dir /data info
docker exec hcpd-server python -m src.server.admin --data-dir /data list-accounts
```

## 3. Tailscale (cho máy khác vào)

Trên máy chạy server (Arch/CachyOS):

```fish
sudo pacman -S tailscale
sudo systemctl enable --now tailscaled
sudo tailscale up
tailscale ip -4
```

Lệnh cuối in địa chỉ `100.x.y.z`. Làm cert **có địa chỉ đó** (cert phải biết địa chỉ nó phục vụ),
nạp lại, đặt địa chỉ vào `deploy/.env` (`HCPD_BIND=100.x.y.z`) rồi chạy lại:

```fish
rm -r ~/hcpd-certs
bash tools/make_dev_cert.sh ~/hcpd-certs 100.x.y.z
bash deploy/import_certs.sh ~/hcpd-certs
cd deploy; docker compose up -d; cd ..
```

Cổng giờ chỉ trả lời trên địa chỉ Tailscale (không trả lời trên LAN). Máy người chơi cũng phải
cài Tailscale và được mời vào cùng tailnet của bạn.

## 4. Máy người chơi

Gửi `cert.pem` (không phải `key.pem`) cho họ, đặt vào `assets/config/server_cert.pem`, rồi tạo
`assets/config/central.json` (đường dẫn cert tính từ thư mục chứa file này):

```json
{"mode": "remote", "host": "100.x.y.z", "port": 47830, "cert": "server_cert.pem", "name": "Tên server"}
```

`name` là tên hiện trên bảng xếp hạng ("Players on <name>"; bỏ đi thì hiện `host`). Xoá file này
là về chơi một mình như cũ.

## 5. Cập nhật server

```fish
git pull
cd deploy; docker compose build; docker compose up -d; cd ..
```

Dữ liệu nằm trong volume nên giữ nguyên. Nếu schema đổi, server tự nâng cấp khi khởi động.

## 6. Sao lưu và khôi phục

Service `backup` chạy ngay khi khởi động rồi mỗi 24 giờ (`HCPD_BACKUP_EVERY_S`), dùng API
sao lưu online của SQLite (an toàn khi server đang ghi), kiểm tra bản sao (`integrity_check`),
giữ 7 bản mới nhất, ghi 1 dòng log mỗi lần chạy. Xem: `docker logs hcpd-backup`. Liệt kê:

```fish
docker run --rm -v hcpd-backups:/backups --entrypoint ls hcpd-server -l /backups
```

Khôi phục vào volume **mới** (lệnh từ chối đè lên một DB đang có):

```fish
cd deploy; docker compose stop; cd ..
docker volume create hcpd-data-restored
docker run --rm --user 0 -v hcpd-data-restored:/data --entrypoint chown hcpd-server 10001:10001 /data
docker run --rm -v hcpd-data-restored:/data -v hcpd-backups:/backups:ro --entrypoint python hcpd-server -m src.server.backup restore --data-dir /data /backups/server-YYYYmmdd-HHMMSS.db
```

Rồi đặt `HCPD_DATA_VOLUME=hcpd-data-restored` trong `deploy/.env` và `docker compose up -d`.

## 7. Chuyển sang VM cloud miễn phí

Cùng các file này: cài Docker, lấy mã nguồn về VM, lặp lại mục 1. Làm cert với địa chỉ của VM
(công khai hoặc Tailscale) rồi sửa `central.json` của người chơi. Chuyển dữ liệu cũ bằng cách
chép một file backup sang VM và khôi phục (mục 6) vào volume mới. Chỉ mở cổng 47830 và giữ
`HCPD_BIND` là địa chỉ cần thiết nhất.

## Lưu ý

- Khi cổng đi qua NAT của Docker, server có thể thấy mọi kết nối đến từ cùng một địa chỉ
  gateway; giới hạn theo IP (`HCPD_MAX_CONNS_PER_IP`) khi đó áp cho cả nhóm. Với vài người chơi là đủ.
- Chưa có giám sát. Kiểm tra `docker ps` hiện `(healthy)`.
