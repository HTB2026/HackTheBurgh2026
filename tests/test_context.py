"""CaseContext.waiting: a module waiting on another must not hold its worker slot."""

import asyncio

from earwitness.modules.base import CaseContext


def _ctx(gate: asyncio.Semaphore | None) -> CaseContext:
    ctx = CaseContext("t", "/nonexistent.mp4", "/tmp", {})
    ctx._gate = gate

    return ctx


def test_waiting_lends_the_slot_out_with_one_worker():
    async def scenario() -> list[str]:
        gate = asyncio.Semaphore(1)
        ctx = _ctx(gate)
        order: list[str] = []

        async def early() -> None:  # like geo: starts first, needs "clue" later
            async with gate:
                order.append("early started")

                async with ctx.waiting():
                    order.append(f"early got {await ctx.wait_for('clue', 5)}")

                order.append("early finished")

        async def clue() -> None:  # needs the only slot to run at all
            await asyncio.sleep(0)

            async with gate:
                ctx.publish("clue", "hum")
                order.append("clue published")

        await asyncio.wait_for(asyncio.gather(early(), clue()), 5)

        return order

    assert asyncio.run(scenario()) == [
        "early started",
        "clue published",
        "early got hum",
        "early finished",
    ]


def test_waiting_takes_the_slot_back():
    async def scenario() -> int:
        gate = asyncio.Semaphore(2)

        async with gate:
            async with _ctx(gate).waiting():
                pass

            # Still held by us inside `async with gate`: one of two free.
            return gate._value

    assert asyncio.run(scenario()) == 1


def test_waiting_without_a_gate_is_a_no_op():
    async def scenario() -> None:
        async with _ctx(None).waiting():
            pass

    asyncio.run(scenario())
