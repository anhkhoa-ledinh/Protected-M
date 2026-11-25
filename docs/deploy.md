# Chạy server trung tâm bằng Docker

Một container (server, tự sao lưu) và một script để điều khiển nó. Mọi lệnh chạy từ fish, ở thư mục
nào cũng được:

```fish
bash deploy/server.sh help
```

Cần Docker dùng được bằng user thường (`docker ps` chạy được). Docker Desktop: bật "Start Docker
Desktop when you sign in" để server tự lên khi máy khởi động. Docker Engine: `systemctl enable docker`.

## 1. Lần đầu

```fish
bash deploy/server.sh setup
bash deploy/server.sh start
```

`setup` tạo `deploy/.env` (từ `.env.example`), làm cert + key trong `~/hcpd-certs` (bằng `python3 tools/make_dev_cert.py`, cần `openssl` trên PATH), build image, nạp
cert vào volume. `start` chạy server, đợi đến khi khoẻ, rồi in địa chỉ và đường dẫn `cert.pem`.
Mặc định server chỉ nghe trên `127.0.0.1` (chính máy này). `key.pem` là bí mật, không bao giờ gửi đi.

## 2. Hằng ngày

| lệnh | việc |
|---|---|
| `bash deploy/server.sh status` | chạy chưa, khoẻ chưa, địa chỉ, uptime, bản sao lưu gần nhất, số tài khoản |
| `bash deploy/server.sh logs 100` | 100 dòng log cuối (mặc định 50) |
| `bash deploy/server.sh stop` | dừng (dữ liệu giữ nguyên) |
| `bash deploy/server.sh start` | chạy lại |
| `bash deploy/server.sh update` | sau `git pull`: build lại image, tạo lại container, dữ liệu giữ nguyên |

Server tự chạy lại khi tiến trình tự chết hoặc khi máy khởi động lại. Lưu ý: Docker coi
`docker kill` / `docker stop` là do bạn tự dừng nên **không** tự bật lại; dùng `start`.

## 3. Sao lưu và khôi phục

Server tự sao lưu khi khởi động rồi mỗi 24 giờ (`HCPD_BACKUP_EVERY_S` trong `deploy/.env`), giữ 7
bản (`HCPD_BACKUP_KEEP`), trong volume `hcpd-backups` (tách khỏi volume dữ liệu).

```fish
bash deploy/server.sh backup
bash deploy/server.sh backups
bash deploy/server.sh restore server-20261125-120000.db
bash deploy/server.sh restore server-20261125-120000.db --force
```

`restore` từ chối khi đang có dữ liệu. Với `--force`, nó sao lưu dữ liệu hiện tại trước rồi mới thay.

## 4. Tailscale (máy khác vào được)

Cài và đăng nhập Tailscale (Arch/CachyOS: `pacman -S tailscale`, `systemctl enable --now tailscaled`,
`tailscale up`), rồi lấy địa chỉ:

```fish
tailscale ip -4
```

Cert phải ghi đúng địa chỉ này. Làm lại cert, rồi đặt `HCPD_BIND=100.x.y.z` trong `deploy/.env`:

```fish
rm -r ~/hcpd-certs
bash deploy/server.sh setup 100.x.y.z
nano deploy/.env
bash deploy/server.sh start
```

Máy người chơi cũng phải cài Tailscale và vào cùng tailnet. Cert tự ký có hiệu lực 825 ngày; hết hạn
thì làm lại cert như trên và gửi `cert.pem` mới cho mọi người.

## 5. Máy người chơi

Gửi `cert.pem` (không phải `key.pem`). Đặt vào `assets/config/server_cert.pem` rồi tạo
`assets/config/central.json` (đường dẫn cert tính từ thư mục chứa file này):

```json
{"mode": "remote", "host": "100.x.y.z", "port": 47830, "cert": "server_cert.pem", "name": "Tên server"}
```

`name` hiện trên bảng xếp hạng ("Players on <name>"). Xoá file này là về chơi một mình.

## 6. Chuyển sang máy khác

Máy mới cần Docker và mã nguồn của repo.

```fish
bash deploy/server.sh export ~/hcpd-move
```

Thư mục này có image, bản sao lưu mới, `cert.pem`, `key.pem` và `.env`. **`key.pem` là bí mật**: chuyển
riêng tư, dùng xong thì xoá. Trên máy mới:

```fish
bash deploy/server.sh import ~/hcpd-move
```

`import` nạp image, cert, `.env`, khôi phục bản sao lưu rồi chạy server. Cert giữ nguyên nên người chơi
không phải đổi gì (trừ địa chỉ trong `central.json` nếu địa chỉ đổi và cert không ghi địa chỉ mới).

## Lưu ý

- Qua NAT của Docker, server có thể thấy mọi kết nối đến từ cùng một địa chỉ; giới hạn theo IP
  (`HCPD_MAX_CONNS_PER_IP`) khi đó áp cho cả nhóm. Vài người chơi là đủ.
- Chưa có giám sát: dùng `status`.
