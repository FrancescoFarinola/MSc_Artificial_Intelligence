import Web3 from "web3";


const getWeb3 = () =>
  new Promise((resolve, reject) => {
    // Wait for loading completion to avoid race conditions with web3 injection timing.
    window.addEventListener("load", async () => {
      // Modern dapp browsers...
      if (window.ethereum) {
        
        try {
          console.log("window.ethereum")
        const web3 = new Web3(window.ethereum);

        let accounts = await window.ethereum.request({ method: 'eth_accounts' })
        if(accounts.length === 0){
          reject("Select an account")
        }
        
        let reqAccounts = await window.ethereum.request({ method: 'eth_requestAccounts' })
        
          // Accounts now exposed
          resolve({web3, ethereum: window.ethereum, accounts: reqAccounts});
        } catch (error) {
          reject(error)
          
        }
      }
      // Fallback to localhost; use dev console port by default...
      else {
        reject("No web3 instance injected, using Local web3. Or install MetaMask")
      }
    });
  });

export default getWeb3;