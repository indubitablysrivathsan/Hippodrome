import { useState, useEffect } from "react";
import { meetings } from "../api/client";
import { useApi, Table, Status, Section, Field, Btn, toRows } from "../shared";

export default function MeetingsPage() {
  const list = useApi(meetings.list);
  const detail = useApi(meetings.get);
  const raceList = useApi(meetings.races);

  const [date, setDate] = useState("");
  const [venueId, setVenueId] = useState("");

  useEffect(() => { list.run(); }, []);

  return (
    <div>
      <h2>Meetings</h2>

      <Section title="List all meetings">
        <Btn onClick={() => list.run()}>Refresh</Btn>
        <Status {...list} />
        {list.data && <Table rows={toRows(list.data)} />}
      </Section>

      <Section title="Get meeting (date + venue)">
        <Field label="Date" value={date} onChange={setDate} placeholder="YYYY-MM-DD" type="date" />
        <Field label="Venue ID" value={venueId} onChange={setVenueId} placeholder="e.g. 1" />
        <Btn onClick={() => detail.run(date, venueId)} disabled={!date || !venueId}>Fetch</Btn>
        <Status {...detail} />
        {detail.data && <Table rows={toRows(detail.data)} />}
      </Section>

      <Section title="Races at meeting">
        <Field label="Date" value={date} onChange={setDate} placeholder="YYYY-MM-DD" type="date" />
        <Field label="Venue ID" value={venueId} onChange={setVenueId} placeholder="e.g. 1" />
        <Btn onClick={() => raceList.run(date, venueId)} disabled={!date || !venueId}>Fetch</Btn>
        <Status {...raceList} />
        {raceList.data && <Table rows={toRows(raceList.data)} />}
      </Section>
    </div>
  );
}
