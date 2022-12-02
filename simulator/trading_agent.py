class Trading:
   def __init__(self):
      # initial setting
      self.initial_balance = 10000000

      # market parameter
      self.fee = 0.003 / 100 * 0

      # temporal variables
      self.cash = self.initial_balance

      self.long_inventory = 0
      self.short_inventory = 0
      self.long_price = 0
      self.short_price = 0

      self.balance = 0

      self.index_history = []
      self.balance_history = []
      self.position_history = []
      self.day_start = [0]

   def long(self, price):
      amount = round(self.balance / price)
      self.long_inventory += amount
      self.cash += -(amount * price) + (amount * price * self.fee)
      self.long_price = price

   def short(self, price):
      amount = round(self.balance / price)
      self.short_inventory += amount
      self.cash += (amount * price) - (amount * price * self.fee)
      self.short_price = price

   def exit_long(self, price):
      self.cash += (self.long_inventory * price) - (self.long_inventory * price * self.fee)
      self.long_inventory = 0

   def exit_short(self, price):
      self.cash += -(self.short_inventory * price) - (self.short_inventory * price * self.fee)
      self.short_inventory = 0

   def evaluate_balance(self, price):
      self.balance = self.cash + self.long_inventory * price - self.short_inventory * price
      self.balance_history.append(self.balance)
      self.index_history.append(price)

      if self.long_inventory == 0 and self.short_inventory == 0:
         self.position_history.append(-1)
      elif self.long_inventory > 0:
         self.position_history.append(1)
      elif self.short_inventory > 0:
         self.position_history.append(0)


   def get_result(self):
      return self.index_history, self.balance_history
