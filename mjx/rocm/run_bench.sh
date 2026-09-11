#!/usr/bin/env bash
# One-shot MJX humanoid benchmark on a DigitalOcean AMD GPU Droplet.
# Creates, provisions, benchmarks, retrieves results, and ALWAYS destroys.
#
#   ./run_bench.sh            # full run, then destroy
#   ./run_bench.sh --keep     # leave the Droplet up for debugging
#   MAX_SECONDS=900 ./run_bench.sh
#
# Hard-won design notes, each from a failure that cost real money:
#   * The kill-switch uses an ABSOLUTE wall-clock deadline, not `sleep N`.
#     `sleep` does not advance while a laptop is suspended, so a 30-minute
#     timer survived 2h18m of suspend and never fired. A Droplet billed for
#     three hours as a result.
#   * The run re-execs under systemd-inhibit so the machine cannot suspend
#     mid-benchmark and drop the SSH session.
#   * The benchmark runs DETACHED on the Droplet, so a local disconnect no
#     longer kills the work. We poll for a completion marker.
#   * Teardown retrieves results BEFORE deleting. Deleting first destroys the
#     only copy of any partial data.
#   * Preflight refuses to start if a Droplet is already running.
set -euo pipefail

NAME="${NAME:-mjx-bench}"
SIZE="${SIZE:-gpu-mi355x1-288gb-spot}"
REGION="${REGION:-mem1}"
IMAGE="${IMAGE:-gpu-amd-base}"
ARCH="${ARCH:-gfx950}"
RATE_PER_HOUR="${RATE_PER_HOUR:-4.50}"
MAX_SECONDS="${MAX_SECONDS:-1800}"
BATCHES="${BATCHES:-8192,16384,32768,65536,131072,262144}"
STEPS="${STEPS:-200}"
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15
          -o ServerAliveCountMax=4 -o ConnectTimeout=15)

# Block suspend for the duration. This is the single most important guard:
# a suspended laptop cannot tear down a remote Droplet.
if [ -z "${INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export INHIBITED=1
  exec systemd-inhibit --what=sleep:idle --who="run_bench" \
    --why="GPU Droplet benchmark in flight" -- "$0" "$@"
fi
[ -n "${INHIBITED:-}" ] || echo "WARNING: systemd-inhibit unavailable; do NOT let this machine sleep."

START=$(date +%s)
DEADLINE=$(( START + MAX_SECONDS ))
WATCHDOG_PID=""
IP=""
say() { echo "[+$(( $(date +%s) - START ))s] $*"; }
cost() {
  local s=$(( $(date +%s) - START ))
  python3 -c "print(f'elapsed {$s}s  approx \${$s/3600*$RATE_PER_HOUR:.2f}')"
}

destroy() {
  local rc=$?
  echo
  [ -n "$WATCHDOG_PID" ] && { kill -- -"$WATCHDOG_PID" 2>/dev/null || kill "$WATCHDOG_PID" 2>/dev/null || true; }

  # Salvage whatever exists before deleting. Costs seconds, saves the data.
  if [ -n "$IP" ]; then
    say "salvaging any results before teardown"
    timeout 60 scp -q "${SSH_OPTS[@]}" root@"$IP":~/mjx_bench_results.json ./mi355x_results.json 2>/dev/null \
      && say "recovered mjx_bench_results.json" || say "no results file to recover"
    timeout 30 scp -q "${SSH_OPTS[@]}" root@"$IP":~/bench.log ./bench_remote.log 2>/dev/null || true
  fi

  if [ "$KEEP" = "1" ]; then
    say "--keep set. Droplet LEFT RUNNING and still billing."
    echo "Destroy with: doctl compute droplet delete $NAME --force"
  else
    say "destroying Droplet"
    doctl compute droplet delete "$NAME" --force 2>/dev/null || true
    local gone=0
    for _ in $(seq 1 20); do
      doctl compute droplet list --no-header --format Name 2>/dev/null | grep -qx "$NAME" || { gone=1; break; }
      sleep 3
    done
    [ "$gone" = "1" ] && say "destroyed, meter stopped" \
      || echo "!!! STILL PRESENT. Destroy by hand NOW: doctl compute droplet delete $NAME --force"
  fi
  cost
  exit $rc
}
trap destroy EXIT INT TERM

# --- Preflight -------------------------------------------------------------
say "preflight"
EXISTING=$(doctl compute droplet list --no-header --format Name 2>/dev/null | grep -c . || true)
if [ "$EXISTING" != "0" ]; then
  echo "REFUSING TO START: $EXISTING Droplet(s) already exist and are billing:"
  doctl compute droplet list
  echo "Destroy them first, or set NAME to something unique."
  KEEP=1   # do not let the trap delete someone else's Droplet
  exit 1
fi
KEY=$(doctl compute ssh-key list --no-header --format FingerPrint | head -1)
[ -n "$KEY" ] || { echo "No SSH key on the account."; KEEP=1; exit 1; }

# --- Create ----------------------------------------------------------------
say "creating $SIZE in $REGION (spot: can be reclaimed at any time)"
doctl compute droplet create "$NAME" \
  --size "$SIZE" --image "$IMAGE" --region "$REGION" \
  --ssh-keys "$KEY" --wait >/dev/null

# Kill-switch: absolute deadline, polled. Survives suspend, unlike `sleep N`.
WD=$(mktemp)
cat > "$WD" <<WDEOF
#!/usr/bin/env bash
while [ "\$(date +%s)" -lt $DEADLINE ]; do sleep 20; done
doctl compute droplet delete "$NAME" --force
WDEOF
chmod +x "$WD"
setsid "$WD" >/dev/null 2>&1 &
WATCHDOG_PID=$!
say "kill-switch armed: absolute deadline $(date -d "@$DEADLINE" '+%H:%M:%S')"

IP=$(doctl compute droplet get "$NAME" --format PublicIPv4 --no-header)
say "droplet up at $IP, waiting for sshd"
for _ in $(seq 1 60); do
  ssh "${SSH_OPTS[@]}" root@"$IP" true 2>/dev/null && break
  sleep 5
done

say "copying files"
scp -q "${SSH_OPTS[@]}" setup_rocm.sh bench_mjx_humanoid.py root@"$IP":~/

say "provisioning (ROCm + JAX)"
ssh "${SSH_OPTS[@]}" root@"$IP" "ARCH=$ARCH bash ~/setup_rocm.sh"

# --- Benchmark, detached so a local disconnect cannot kill it --------------
say "launching sweep detached (batches $BATCHES, fixed $STEPS steps)"
ssh "${SSH_OPTS[@]}" root@"$IP" "rm -f ~/bench.done ~/bench.log; \
  setsid nohup bash -c 'source ~/mjx-env/bin/activate && \
    python -u bench_mjx_humanoid.py --batch-sizes $BATCHES --steps $STEPS \
    > ~/bench.log 2>&1; echo \$? > ~/bench.done' >/dev/null 2>&1 &" || true

say "polling for completion"
while :; do
  now=$(date +%s)
  [ "$now" -ge "$DEADLINE" ] && { echo "DEADLINE REACHED"; break; }
  DONE=$(ssh "${SSH_OPTS[@]}" root@"$IP" "cat ~/bench.done 2>/dev/null" 2>/dev/null || echo "")
  [ -n "$DONE" ] && { say "sweep finished (exit $DONE)"; break; }
  sleep 20
done

ssh "${SSH_OPTS[@]}" root@"$IP" "cat ~/bench.log 2>/dev/null" 2>/dev/null || true
say "run complete"
