import { useState, useEffect } from "react";
import { venues } from "../api/client";
import { useApi, Table, Status, Section, Field, Btn, toRows } from "../shared";

export default function VenuesPage() {
  const list = useApi(venues.list);
  const detail = useApi(venues.get);
  const mtgs = useApi(venues.meetings);

  const [id, setId] = useState("");
  const [mtgId, setMtgId] = useState("");

  useEffect(() => { list.run(); }, []);

  return (
    <div>
      <h2>Venues</h2>

      <Section title="List all venues">
        <Btn onClick={() => list.run()}>Refresh</Btn>
        <Status {...list} />
        {list.data && <Table rows={toRows(list.data)} />}
      </Section>

      <Section title="Get venue by ID">
        <Field label="Venue ID" value={id} onChange={setId} placeholder="e.g. 1" />
        <Btn onClick={() => detail.run(id)} disabled={!id}>Fetch</Btn>
        <Status {...detail} />
        {detail.data && <Table rows={toRows(detail.data)} />}
      </Section>

      <Section title="Venue meetings">
        <Field label="Venue ID" value={mtgId} onChange={setMtgId} placeholder="e.g. 1" />
        <Btn onClick={() => mtgs.run(mtgId)} disabled={!mtgId}>Fetch</Btn>
        <Status {...mtgs} />
        {mtgs.data && <Table rows={toRows(mtgs.data)} />}
      </Section>
    </div>
  );
}
