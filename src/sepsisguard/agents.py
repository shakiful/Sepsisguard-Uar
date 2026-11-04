"""Bounded LangGraph retrieval/draft/verification loop with fail-closed outputs."""
from __future__ import annotations

from collections import Counter
import json
import os
import re
from typing import TypedDict

from .evidence import normalize


class AgentState(TypedDict):
    case: dict
    attempts: int
    calls: int
    query: str
    passages: list
    claims: list
    status: str
    reason: str
    trace: list


ALLOWED_CASE_KEYS={'case_id','raw_score','calibrated_risk','predictive_entropy_bits','observation_missing_fraction','observations','calibration_fitted'}
BLOCKED=re.compile(r'\b(?:patient|case)\b.{0,30}\b(?:has|is|confirms?)\b.{0,20}\bsepsis\b|\b(administer|prescribe|dosage|diagnose|diagnosed)\b|mg/kg|ml/kg|\b(give|start)\b.{0,30}\b(antibiotics?|fluids?|treatment)\b',re.I)
RISK_NUMBER=re.compile(r'\b(risk|probability|confidence|chance)\b.{0,35}(?:\d|percent|%)',re.I)


def sanitize_case(case):
    if not isinstance(case,dict) or set(case)-ALLOWED_CASE_KEYS:
        raise ValueError('Unsupported case fields; identifiers, labels and future data are excluded')
    if case.get('calibration_fitted') is not True:
        raise ValueError('A fitted held-out calibrator is required for risk routing')
    for key in ('raw_score','calibrated_risk','predictive_entropy_bits','observation_missing_fraction'):
        if not isinstance(case.get(key),(float,int)) or not 0<=case[key]<=1:
            raise ValueError('Invalid numerical case field')
    if not isinstance(case.get('observations'),dict):
        raise ValueError('Observed endpoint measurements are required')
    if not set(case['observations']) <= {'HR','Resp','Temp','MAP','Lactate','O2Sat','SBP','DBP'}:
        raise ValueError('Unexpected observation channel')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not __import__('math').isfinite(v) for v in case['observations'].values()):
        raise ValueError('Only finite observed numbers enter prompts')
    return dict(case)


def citation_checks(claims,passages,case):
    errors=[];quoted=Counter();by_id={p['passage_id']:p for p in passages}
    if not isinstance(claims,list) or not 1<=len(claims)<=2:
        return ['Expected one or two structured claims']
    for claim in claims:
        if not isinstance(claim,dict) or set(claim)!={'text','citations'}:
            errors.append('Invalid claim schema');continue
        text=claim['text'];cites=claim['citations']
        if not isinstance(text,str) or not 5<=len(text)<=600 or BLOCKED.search(text) or RISK_NUMBER.search(text):
            errors.append('Unsupported claim content');continue
        if not isinstance(cites,list) or not 1<=len(cites)<=2:
            errors.append('Claim lacks bounded citations');continue
        evidence=[]
        for citation in cites:
            if not isinstance(citation,dict) or set(citation)!={'passage_id','quote'} or not isinstance(citation.get('passage_id'),str) or citation['passage_id'] not in by_id:
                errors.append('Unknown evidence citation');continue
            p=by_id[citation['passage_id']];quote=citation['quote']
            if not isinstance(quote,str) or not 3<=len(quote.split())<=25 or normalize(quote) not in normalize(p['text']):
                errors.append('Quote not present in cited passage');continue
            quoted[p['source_id']]+=len(quote.split());evidence.append(p['text'])
        allowed=set(re.findall(r'\d+(?:\.\d+)?',' '.join(evidence)+' '+json.dumps(case['observations'])))
        if set(re.findall(r'\d+(?:\.\d+)?',text))-allowed:
            errors.append('Unverified number in claim')
    if any(n>25 for n in quoted.values()):errors.append('Per-source quote budget exceeded')
    return errors


class ResearchAgent:
    def __init__(self,llm,retriever,*,max_calls=6,max_attempts=2):
        if not 3<=max_calls<=6 or not 1<=max_attempts<=2:
            raise ValueError('Agent budgets exceed the implemented bounds')
        self.llm,self.retriever=llm,retriever
        self.max_calls,self.max_attempts=max_calls,max_attempts

    def _call(self,state,role,payload):
        if state['calls']>=self.max_calls:
            raise ValueError('Tool-decision budget exhausted')
        state['calls']+=1
        result=self.llm.json(role,payload)
        if not isinstance(result,dict):raise ValueError('LLM must return a JSON object')
        state['trace'].append({'role':role,'response':result})
        return result

    def _abstain(self,state,reason):
        state['status']='abstained';state['reason']=reason
        return state

    def _plan(self,state):
        if state['attempts']>=self.max_attempts:
            return self._abstain(state,'Retrieval budget exhausted')
        try:
            plan=self._call(state,'planner',{'case':state['case'],'previous_failure':state['reason']})
            if plan.get('action')=='abstain':return self._abstain(state,'Planner found insufficient evidence')
            if plan.get('action')!='search' or set(plan)!={'action','query'}:
                return self._abstain(state,'Disallowed planner action')
            if not isinstance(plan['query'],str) or not 2<=len(plan['query'])<=300:
                return self._abstain(state,'Invalid retrieval query')
            state['query']=plan['query'];state['status']='search'
        except (ValueError,RuntimeError,TimeoutError) as e:
            return self._abstain(state,f'Planner failed: {type(e).__name__}')
        return state

    def _retrieve(self,state):
        state['attempts']+=1
        state['passages']=self.retriever.search(state['query'])
        state['trace'].append({'tool':'local_retrieval','query':state['query'],
                               'passage_ids':[p['passage_id'] for p in state['passages']]})
        if not state['passages']:return self._abstain(state,'No relevant reference passages')
        state['status']='draft';return state

    def _draft(self,state):
        try:
            response=self._call(state,'writer',{'case':state['case'],'evidence':state['passages']})
            if response.get('action')=='abstain':return self._abstain(state,'Writer found insufficient evidence')
            state['claims']=response.get('claims')
            errors=citation_checks(state['claims'],state['passages'],state['case'])
            if errors:
                state['reason']='; '.join(errors);state['status']='retry';return state
            state['status']='verify'
        except (ValueError,RuntimeError,TimeoutError) as e:
            return self._abstain(state,f'Draft failed: {type(e).__name__}')
        return state

    def _verify(self,state):
        try:
            decision=self._call(state,'verifier',{'case':state['case'],'claims':state['claims'],'evidence':state['passages']})
            decisions=decision.get('decisions')
            valid=(isinstance(decisions,list) and len(decisions)==len(state['claims'])
                   and all(isinstance(x,dict) and set(x)=={'claim_index','supported'}
                           and type(x['claim_index']) is int and x['supported'] is True for x in decisions))
            if not valid or {x['claim_index'] for x in decisions}!=set(range(len(state['claims']))):
                state['status']='retry';state['reason']='Verifier did not support every claim'
            else:
                state['status']='supported_research_context';state['reason']='Citation checks and model entailment check passed'
        except (ValueError,RuntimeError,TimeoutError) as e:
            return self._abstain(state,f'Verifier failed: {type(e).__name__}')
        return state

    def _after(self,state):
        if state['status']=='retry':
            if state['attempts']<self.max_attempts and state['calls']+3<=self.max_calls:return 'plan'
            return 'finish'
        return {'search':'retrieve','draft':'draft','verify':'verify'}.get(state['status'],'finish')

    def run(self,case):
        os.environ['LANGSMITH_TRACING']='false'
        os.environ['LANGCHAIN_TRACING_V2']='false'
        from langgraph.graph import StateGraph,START,END
        case=sanitize_case(case)
        graph=StateGraph(AgentState)
        for name,node in [('plan',self._plan),('retrieve',self._retrieve),('draft',self._draft),('verify',self._verify)]:
            graph.add_node(name,node)
            graph.add_conditional_edges(name,self._after,{'plan':'plan','retrieve':'retrieve','draft':'draft','verify':'verify','finish':END})
        graph.add_edge(START,'plan')
        # Predictive entropy is a routing heuristic, not a correctness probability.
        attempts=self.max_attempts if case['predictive_entropy_bits']>=0.3 or case['observation_missing_fraction']>=0.5 else 1
        original=self.max_attempts;self.max_attempts=attempts
        try:
            state=graph.compile().invoke({'case':case,'attempts':0,'calls':0,'query':'','passages':[],
                                          'claims':[],'status':'pending','reason':'','trace':[]},config={'recursion_limit':30})
        finally:self.max_attempts=original
        if state['status']!='supported_research_context':
            state['status']='abstained';answer=None
        else:
            answer=' '.join(c['text']+' ['+', '.join(x['passage_id'] for x in c['citations'])+']' for c in state['claims'])
        return {**state,'answer':answer,'verification_scope':'Programmatic citation/numeric checks plus same-model entailment; not independent factual certification',
                'intended_use':'Retrospective research explanation; no diagnosis or treatment recommendation'}
