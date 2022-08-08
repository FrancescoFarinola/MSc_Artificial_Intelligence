import React from 'react';
import { createContext, useState } from "react";

/*
If we have three components in our app, A->B->C where A is the parent of B and B is the parent of C. 
To change a state from C and pass it to A, keep the state of A in a store, then extract the state from 
store and use it in C. This completely eliminates the necessity of the state to pass through B. So the flow is like A->C.
*/

const RoleContext = createContext(null);

export const RoleContextProvider = ({ fRole, mRole, dRole, rRole, cRole, children }) => {

    const [roles, setRoles] = useState({
        farmer : fRole,
        manufacturer : mRole,
        distributor : dRole,
        retailer : rRole,
        consumer : cRole
    });

  return (
    <RoleContext.Provider value={{ roles, setRoles }}>
      {children}
    </RoleContext.Provider>
  );
};

export const useRole = () => React.useContext(RoleContext);