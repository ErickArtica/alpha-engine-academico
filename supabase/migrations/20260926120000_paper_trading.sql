-- Isolated virtual portfolio. Only the trusted backend may read or write it.
-- Reapplying this migration is safe; no market/prediction tables are changed.
create table if not exists public.paper_accounts (
    portfolio_id uuid primary key,
    initial_capital numeric(24, 8) not null check (initial_capital > 0),
    allocation numeric(12, 8) not null check (allocation > 0 and allocation <= 1),
    fee_rate numeric(12, 8) not null check (fee_rate >= 0 and fee_rate < 1),
    slippage_rate numeric(12, 8) not null check (slippage_rate >= 0 and slippage_rate < 1),
    cash numeric(24, 8) not null check (cash >= 0),
    btc numeric(28, 16) not null default 0 check (btc >= 0),
    entry_price numeric(24, 8),
    entry_cost numeric(24, 8),
    realized_pnl numeric(24, 8) not null default 0,
    fees_paid numeric(24, 8) not null default 0,
    closed_trades integer not null default 0,
    winning_trades integer not null default 0,
    operation_count integer not null default 0,
    generation integer not null default 0,
    version integer not null default 0,
    last_reset_id uuid,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check ((btc = 0 and entry_price is null and entry_cost is null)
           or (btc > 0 and entry_price > 0 and entry_cost > 0))
);
alter table public.paper_accounts add column if not exists operation_count integer not null default 0;

create table if not exists public.paper_trades (
    trade_id uuid primary key,
    portfolio_id uuid not null references public.paper_accounts(portfolio_id),
    generation integer not null,
    prediction_id text not null,
    executed_at timestamptz not null,
    action text not null check (action in ('BUY', 'SELL')),
    reference_price numeric(24, 8) not null check (reference_price > 0),
    price numeric(24, 8) not null check (price > 0),
    quantity_btc numeric(28, 16) not null check (quantity_btc > 0),
    value_usd numeric(24, 8) not null check (value_usd > 0),
    fee numeric(24, 8) not null check (fee >= 0),
    cash_after numeric(24, 8) not null check (cash_after >= 0),
    btc_after numeric(28, 16) not null check (btc_after >= 0),
    realized_pnl numeric(24, 8),
    model_version text not null,
    signal_probability numeric(12, 8) not null check (signal_probability between 0 and 1),
    unique (portfolio_id, generation, prediction_id)
);
create index if not exists paper_trades_recent
    on public.paper_trades (portfolio_id, generation, executed_at desc);

create or replace function public.paper_apply(
    p_portfolio_id uuid, p_trade_id uuid, p_prediction_id text,
    p_action text, p_reference_price numeric, p_model_version text,
    p_signal_probability numeric, p_expected_version integer,
    p_executed_at timestamptz
) returns text language plpgsql set search_path = '' as $$
declare
    a public.paper_accounts%rowtype;
    v_price numeric;
    v_quantity numeric;
    v_value numeric;
    v_fee numeric;
    v_pnl numeric;
begin
    select * into a from public.paper_accounts
    where portfolio_id = p_portfolio_id for update;
    if not found then return 'NOT_FOUND'; end if;
    if exists (select 1 from public.paper_trades where trade_id = p_trade_id
               and portfolio_id = p_portfolio_id) then return 'DUPLICATE'; end if;
    if a.version <> p_expected_version then return 'CONFLICT'; end if;
    if exists (select 1 from public.paper_trades where portfolio_id = p_portfolio_id
               and generation = a.generation and prediction_id = p_prediction_id) then
        return 'DUPLICATE_PREDICTION';
    end if;
    if p_reference_price <= 0 or p_signal_probability < 0 or p_signal_probability > 1
       or p_prediction_id is null or p_model_version is null or p_executed_at is null then
        return 'INVALID_REQUEST';
    end if;
    if p_action = 'BUY' and a.btc = 0 then
        v_price := p_reference_price * (1 + a.slippage_rate);
        v_value := a.cash * a.allocation / (1 + a.fee_rate);
        v_fee := v_value * a.fee_rate;
        v_quantity := v_value / v_price;
        if v_quantity <= 0 then return 'INVALID_REQUEST'; end if;
        update public.paper_accounts set cash = cash - v_value - v_fee,
            btc = v_quantity, entry_price = v_price, entry_cost = v_value + v_fee,
            fees_paid = fees_paid + v_fee, operation_count = operation_count + 1,
            version = version + 1,
            updated_at = p_executed_at where portfolio_id = p_portfolio_id;
        v_pnl := null;
    elsif p_action = 'SELL' and a.btc > 0 then
        v_price := p_reference_price * (1 - a.slippage_rate);
        v_quantity := a.btc;
        v_value := v_quantity * v_price;
        v_fee := v_value * a.fee_rate;
        v_pnl := v_value - v_fee - a.entry_cost;
        update public.paper_accounts set cash = cash + v_value - v_fee,
            btc = 0, entry_price = null, entry_cost = null,
            realized_pnl = realized_pnl + v_pnl, fees_paid = fees_paid + v_fee,
            closed_trades = closed_trades + 1,
            winning_trades = winning_trades + case when v_pnl > 0 then 1 else 0 end,
            operation_count = operation_count + 1,
            version = version + 1, updated_at = p_executed_at
            where portfolio_id = p_portfolio_id;
    else
        return 'INVALID_ACTION';
    end if;
    select * into a from public.paper_accounts where portfolio_id = p_portfolio_id;
    insert into public.paper_trades (
        trade_id, portfolio_id, generation, prediction_id, executed_at, action,
        reference_price, price, quantity_btc, value_usd, fee, cash_after,
        btc_after, realized_pnl, model_version, signal_probability
    ) values (
        p_trade_id, p_portfolio_id, a.generation, p_prediction_id, p_executed_at,
        p_action, p_reference_price, v_price, v_quantity, v_value, v_fee,
        a.cash, a.btc, v_pnl, p_model_version, p_signal_probability
    );
    return 'APPLIED';
end;
$$;

create or replace function public.paper_reset(
    p_portfolio_id uuid, p_reset_id uuid, p_expected_version integer,
    p_reset_at timestamptz
) returns text language plpgsql set search_path = '' as $$
declare a public.paper_accounts%rowtype;
begin
    select * into a from public.paper_accounts
    where portfolio_id = p_portfolio_id for update;
    if not found then return 'NOT_FOUND'; end if;
    if a.last_reset_id = p_reset_id then return 'DUPLICATE'; end if;
    if a.version <> p_expected_version then return 'CONFLICT'; end if;
    update public.paper_accounts set cash = initial_capital, btc = 0,
        entry_price = null, entry_cost = null, realized_pnl = 0, fees_paid = 0,
        closed_trades = 0, winning_trades = 0, operation_count = 0,
        generation = generation + 1,
        version = version + 1, last_reset_id = p_reset_id, updated_at = p_reset_at
    where portfolio_id = p_portfolio_id;
    return 'RESET';
end;
$$;

alter table public.paper_accounts enable row level security;
alter table public.paper_trades enable row level security;
revoke all on public.paper_accounts, public.paper_trades from public, anon, authenticated;
grant all on public.paper_accounts, public.paper_trades to service_role;
revoke all on function public.paper_apply(uuid,uuid,text,text,numeric,text,numeric,integer,timestamptz)
    from public, anon, authenticated;
revoke all on function public.paper_reset(uuid,uuid,integer,timestamptz)
    from public, anon, authenticated;
grant execute on function public.paper_apply(uuid,uuid,text,text,numeric,text,numeric,integer,timestamptz)
    to service_role;
grant execute on function public.paper_reset(uuid,uuid,integer,timestamptz)
    to service_role;
