"""An independent GEM host for the tool emulator: secsgem (Python) instead of WaferEdge's own.

Connects to a running tool-emulator, receives its S6F11 wafer reports, holds one lot with
S2F41 after that lot's first wafer, and prints what arrived. It checks that WaferEdge's HSMS,
SECS-II and GEM interoperate with a separate implementation (docs/pipeline.md).

    python3 -m venv /tmp/secsgem-venv && /tmp/secsgem-venv/bin/pip install secsgem==0.3.0
    build/release/tools/tool-emulator tests/data/waferlens_sample.wmap --port 5000 --rate 5 &
    /tmp/secsgem-venv/bin/python tools/secsgem_host.py --port 5000 --hold LOT-000281

secsgem decodes S6F11 against the reports it defined itself (S2F33/35); the emulator's report
10 is predefined (ADR-0011), so its five variables are registered here by hand.
"""

import argparse
import threading
import time

import secsgem.common
import secsgem.gem
import secsgem.hsms


def value(item):
    """secsgem hands over items or plain Python values, depending on the format."""
    return item.get() if hasattr(item, "get") else item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--hold", default="LOT-000281", help="lot to hold after its first wafer")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds to wait in total")
    args = parser.parse_args()

    settings = secsgem.hsms.HsmsSettings(
        address="127.0.0.1",
        port=args.port,
        connect_mode=secsgem.hsms.HsmsConnectMode.ACTIVE,
        device_type=secsgem.common.DeviceType.HOST,
    )
    host = secsgem.gem.GemHostHandler(settings)
    host.report_subscriptions[10] = ["LOTID", "WAFERID", "ROWS", "COLS", "BINS"]

    received = []
    hold_sent = threading.Event()
    hold_done = threading.Event()

    def hold() -> None:
        reply = host.send_remote_command("HOLD", [["LOTID", args.hold]])
        print(f"S2F42 HCACK {reply.HCACK.get()} for {args.hold}", flush=True)
        hold_done.set()

    def on_report(data) -> None:
        v = {x["dvid"]: value(x["value"]) for x in data["values"]}
        bins = v["BINS"]
        count = len(bins) if isinstance(bins, (list, tuple)) else 1
        received.append((v["LOTID"], v["WAFERID"], hold_done.is_set()))
        print(f"CEID {value(data['ceid'])}: {v['LOTID']} wafer {v['WAFERID']} "
              f"{v['ROWS']}x{v['COLS']} ({count} bins)", flush=True)
        if v["LOTID"] == args.hold and not hold_sent.is_set():
            hold_sent.set()
            threading.Thread(target=hold).start()  # don't block secsgem's receive thread

    host.events.collection_event_received += on_report
    host.enable()
    deadline = time.time() + args.timeout
    last = 0
    while time.time() < deadline:  # done when the emulator has been quiet for 2 s
        time.sleep(2.0)
        if received and len(received) == last:
            break
        last = len(received)
    host.disable()

    late = [w for (lot, w, after_hold) in received if lot == args.hold and after_hold]
    print(f"{len(received)} wafer reports; {args.hold} wafers after the HCACK: {late or 'none'}")


if __name__ == "__main__":
    main()
